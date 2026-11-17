"""Notices NAF rév. 2 (NAF 2008) au format des notices du graphe, depuis les PDF Insee.

L'Insee ne publie les notes explicatives de la NAF rév. 2 (« comprend », « comprend
aussi », « ne comprend pas ») qu'en PDF ; seuls les intitulés existent en tableur.
Ce script assemble les deux :

- les intitulés (`int_courts_naf_rev_2.xls`) donnent la liste exhaustive et ordonnée des
  postes — 21 sections, 88 divisions, 272 groupes, 615 classes, 732 sous-classes — et
  leurs noms exacts ;
- le PDF (`Nomenclatures_NAF_Reedition_2020.pdf`) donne le texte des notes. Chaque
  poste y ouvre un bloc par une ligne « <code> <intitulé> », dans l'ordre même du
  tableur : on cherche donc, ligne après ligne, le titre du *prochain* poste attendu
  et seulement lui. Une ligne de texte qui commencerait par un autre code ne peut pas
  être prise pour un titre.

Le parquet produit a les colonnes de `Notices-NAF2025-FR.parquet` (cf.
graph_builder.config.COLUMNS_TO_KEEP), avec la même convention pour les classes à
sous-classe unique : en NAF 2025, `01.14` et `01.14Y` ne font qu'un noeud de niveau 4,
final, de code `01.14Y` et d'ID `0114`. On fait de même ici (`01.11` + `01.11Z` →
noeud `01.11Z`, ID `0111`), pour que le graphe NAF 2008 ait la forme que les agents
connaissent déjà : une feuille est un code à 5 caractères, au niveau 4 ou 5.

Sources (fichiers publics) :
    https://www.insee.fr/fr/statistiques/fichier/2120875/Nomenclatures_NAF_Reedition_2020.pdf
    https://www.insee.fr/fr/statistiques/fichier/2120875/int_courts_naf_rev_2.xls

Usage (xlrd n'est utile qu'ici, d'où le `--with`) :
    uv run --with xlrd -m src.neo4j_graph.graph_builder.utils.naf2008_notices \
        --pdf Nomenclatures_NAF_Reedition_2020.pdf --labels int_courts_naf_rev_2.xls \
        --output s3://projet-ape/notices/Notices-NAF2008-FR.parquet
"""

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
import pypdf

logger = logging.getLogger(__name__)

ORDERED_COLUMNS = [
    "ORDER_KEY",
    "ID",
    "CODE",
    "NAME",
    "PARENT_ID",
    "PARENT_CODE",
    "LEVEL",
    "FINAL",
    "Implementation_rule",
    "Includes",
    "IncludesAlso",
    "Excludes",
    "text_content",
]

# Pied de page répété sur chaque page du PDF.
_FOOTER = re.compile(r"^Nomenclature d.Activit[ée]s Fran[çc]aise NAF r[ée]v\. 2 \d+\s*$")
# « Produits associés : 47.00.11p, 47.00.12p, … » et ses lignes de continuation : les
# codes CPF associés ne disent rien du champ du poste et ne feraient que du bruit.
_PRODUCTS_START = re.compile(r"^\s*Produits associés\s*:")
_PRODUCTS_CONT = re.compile(r"^[\s\d.,p\-]+$")
_SUBJECT = r"(?:Cette|Ce|Cet)\s+(?:section|division|groupe|classe|sous-classe)"
_ALSO = re.compile(_SUBJECT + r"\s+comprend\s+aussi")
_EXCLUDES = re.compile(_SUBJECT + r"\s+ne\s+comprend\s+pas")
# Une ligne qui commence par un de ces marqueurs ouvre un nouvel élément ; les autres
# prolongent la ligne précédente (retour à la ligne du PDF).
_NEW_ITEM = re.compile(r"^\s*(?:-\s|•|" + _SUBJECT + r")")


@dataclass
class Post:
    code: str  # forme pointée, telle que dans le tableur (« 01.11Z », « A »)
    name: str
    level: int
    body: list[str] = field(default_factory=list)


def _level_of(code: str) -> int:
    if len(code) == 1:
        return 1
    return {2: 2, 4: 3, 5: 4, 6: 5}[len(code)]


def _norm(text: str) -> str:
    """Forme de comparaison : sans accents, casse, ponctuation ni espaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_posts(labels_path: str) -> list[Post]:
    labels = pd.read_excel(labels_path, dtype=str)
    labels = labels.iloc[:, [1, 2]].dropna()
    labels.columns = ["code", "name"]
    posts = []
    for code, name in zip(labels["code"].str.strip(), labels["name"].str.strip()):
        code = code.removeprefix("SECTION ").strip()
        posts.append(Post(code=code, name=name, level=_level_of(code)))
    return posts


def _heading_prefix(post: Post) -> str:
    return f"Section {post.code} :" if post.level == 1 else f"{post.code} "


def extract_lines(pdf_path: str) -> list[str]:
    reader = pypdf.PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Certaines pages sortent des tabulations ou des espaces insécables entre les mots
    # (« 01.16\tCulture\tde\tplantes ») : sans cela, le titre ne serait pas reconnu.
    lines = (re.sub(r"[ \t  ]+", " ", line) for line in text.splitlines())
    return [line for line in lines if not _FOOTER.match(line)]


def attach_bodies(posts: list[Post], lines: list[str]) -> list[str]:
    """Répartit les lignes du PDF entre les postes ; renvoie les anomalies relevées.

    Le sommaire du PDF reprend sections et divisions : on ne commence qu'au premier
    titre de section suivi de son intitulé en capitales, c'est-à-dire au corps.
    """
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f"Section A : {posts[0].name[:10]}")
    )
    def is_heading(line: str, post: Post) -> bool:
        # Le code seul ne suffit pas : une phrase de notice peut commencer par le code
        # attendu (« 50.1 et 50.2, tandis que… »). Le début de l'intitulé doit suivre.
        prefix = _heading_prefix(post)
        if not line.startswith(prefix):
            return False
        read, expected = _norm(line[len(prefix) :]), _norm(post.name)
        n = min(8, len(read), len(expected))
        return n > 0 and read[:n] == expected[:n]

    issues = []
    i, k, current = start, 0, None
    while i < len(lines):
        line = lines[i].lstrip()
        if k < len(posts) and is_heading(line, posts[k]):
            current = posts[k]
            # L'intitulé peut courir sur plusieurs lignes : on les consomme tant qu'il
            # n'est pas complet, pour ne pas les prendre pour du texte de notice.
            heading = line[len(_heading_prefix(current)) :]
            target = _norm(current.name)
            while len(_norm(heading)) < len(target) and i + 1 < len(lines):
                nxt = lines[i + 1].lstrip()
                if (k + 1 < len(posts) and is_heading(nxt, posts[k + 1])) or _NEW_ITEM.match(nxt):
                    break
                heading += " " + nxt
                i += 1
            if not _norm(heading).startswith(target[:20]):
                issues.append(f"{current.code}: intitulé lu « {heading[:60]} »")
            k += 1
        elif current is not None:
            current.body.append(lines[i])
        i += 1
    if k < len(posts):
        issues.append(f"{len(posts) - k} postes introuvables, à partir de {posts[k].code}")
    return issues


def _join_wrapped(lines: list[str]) -> str:
    """Recolle les lignes coupées par la mise en page, garde les puces sur leur ligne."""
    items: list[str] = []
    skipping_products = False
    for line in lines:
        if _PRODUCTS_START.match(line):
            skipping_products = True
            continue
        if skipping_products and _PRODUCTS_CONT.match(line):
            continue
        skipping_products = False
        stripped = line.strip()
        if not stripped:
            continue
        if not items or _NEW_ITEM.match(line):
            items.append(stripped)
        else:
            items[-1] += " " + stripped
    return "\n".join(items)


def split_sections(body: str) -> tuple[str | None, str | None, str | None]:
    """(Includes, IncludesAlso, Excludes) : dans le PDF, toujours dans cet ordre."""
    excl = _EXCLUDES.search(body)
    also = _ALSO.search(body[: excl.start()] if excl else body)
    end_includes = also.start() if also else (excl.start() if excl else len(body))
    includes = body[:end_includes].strip() or None
    includes_also = (
        body[also.start() : excl.start() if excl else len(body)].strip() if also else None
    )
    excludes = body[excl.start() :].strip() if excl else None
    return includes, includes_also, excludes


def _merge(*texts: str | None) -> str | None:
    kept = [t for t in texts if t]
    return "\n".join(kept) if kept else None


def build_notices(posts: list[Post]) -> pd.DataFrame:
    children: dict[str, list[Post]] = {}
    for post in posts:
        if post.level == 5:
            children.setdefault(post.code[:5], []).append(post)

    section_of: dict[str, str] = {}
    current_section = None
    for post in posts:
        if post.level == 1:
            current_section = post.code
        elif post.level == 2:
            section_of[post.code] = current_section

    def parent_code(post: Post) -> str | None:
        return {
            1: None,
            2: section_of.get(post.code),
            3: post.code[:2],
            4: post.code[:4],
            5: post.code[:5],
        }[post.level]

    rows = []
    by_code = {p.code: p for p in posts}
    for order, post in enumerate(posts):
        sections = split_sections(_join_wrapped(post.body))
        row = {
            "ORDER_KEY": float(order),
            "ID": post.code.replace(".", ""),
            "CODE": post.code,
            "NAME": post.name,
            "PARENT_CODE": parent_code(post),
            "LEVEL": post.level,
            "FINAL": int(post.level == 5),
        }
        if post.level == 4 and len(children.get(post.code, [])) == 1:
            # Classe à sous-classe unique : un seul noeud, cf. docstring du module.
            continue
        if post.level == 5 and len(children[post.code[:5]]) == 1:
            klass = by_code[post.code[:5]]
            class_sections = split_sections(_join_wrapped(klass.body))
            sections = tuple(_merge(a, b) for a, b in zip(class_sections, sections))
            row.update(
                ID=klass.code.replace(".", ""),
                LEVEL=4,
                PARENT_CODE=parent_code(klass),
            )
        row["Includes"], row["IncludesAlso"], row["Excludes"] = sections
        rows.append(row)

    df = pd.DataFrame(rows)
    df["PARENT_ID"] = df["PARENT_CODE"].str.replace(".", "", regex=False)
    df["Implementation_rule"] = None
    df["text_content"] = [
        "\n\n".join(t for t in (name, inc, also, exc) if isinstance(t, str) and t)
        for name, inc, also, exc in zip(
            df["NAME"], df["Includes"], df["IncludesAlso"], df["Excludes"]
        )
    ]
    return df[ORDERED_COLUMNS]


def check(df: pd.DataFrame) -> list[str]:
    """Contrôles structurels : effectifs attendus, parents existants, feuilles."""
    issues = []
    final = df[df["FINAL"] == 1]
    if len(final) != 732:
        issues.append(f"{len(final)} feuilles au lieu de 732 sous-classes")
    if df["CODE"].duplicated().any():
        issues.append(f"codes en double : {df.loc[df['CODE'].duplicated(), 'CODE'].tolist()}")
    if df["ID"].duplicated().any():
        issues.append(f"ID en double : {df.loc[df['ID'].duplicated(), 'ID'].tolist()}")
    orphans = df[df["PARENT_ID"].notna() & ~df["PARENT_ID"].isin(df["ID"])]
    if len(orphans):
        issues.append(f"parents introuvables : {orphans['CODE'].tolist()[:10]}")
    empty = final[final[["Includes", "IncludesAlso", "Excludes"]].isna().all(axis=1)]
    logger.info(f"{len(empty)} feuilles sans aucune note (intitulé seul)")
    return issues


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", required=True, help="Nomenclatures_NAF_Reedition_2020.pdf")
    parser.add_argument("--labels", required=True, help="int_courts_naf_rev_2.xls")
    parser.add_argument("--output", required=True, help="Parquet de sortie, local ou s3://")
    args = parser.parse_args()

    posts = load_posts(args.labels)
    issues = attach_bodies(posts, extract_lines(args.pdf))
    df = build_notices(posts)
    issues += check(df)
    for issue in issues:
        logger.warning(issue)
    logger.info(f"{len(df)} noeuds : {df.groupby('LEVEL').size().to_dict()}")

    if args.output.startswith("s3://"):
        from src.utils.storage import get_file_system

        df.to_parquet(args.output.removeprefix("s3://"), filesystem=get_file_system())
    else:
        df.to_parquet(args.output)
    logger.info(f"Écrit : {args.output}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
