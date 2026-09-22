"""Codes candidats proposés à l'annotateur, précalculés pour un run de match_verifier_eval.

L'app de revue (cf. src.evaluation.apps.match_verifier_eval_app) demande à l'annotateur,
quand il juge le code faux, quel serait le bon — jusqu'ici dans un champ libre, qui
suppose de connaître la nomenclature de tête. Ce script prépare, pour chaque ligne du
parquet d'un run, une courte liste de codes plausibles que l'app affiche en regard du
champ : l'annotateur clique au lieu de chercher.

Trois sources indépendantes, fusionnées :

- **`supervised`** — le modèle supervisé de production via codif-ape-API (cf.
  SupervisedClassifier.top_k). C'est la seule dont le score soit une probabilité
  calibrée, et le seul candidat qui ne partage aucun composant avec le MatchVerifier.
- **`embedding`** — la recherche par similarité sur les notices indexées dans Neo4j
  (cf. Graph.get_closest_codes), celle-là même dont part l'AgenticRAGClassifier.
- **`alternative`** — le meilleur code concurrent que le MatchVerifier a lui-même nommé
  sur cette ligne (colonne `match_verifier_alternative_code`), déjà dans le parquet :
  gratuit, et c'est le candidat que l'annotateur voit de toute façon plus haut dans la
  page.

Les trois n'ont pas d'échelle commune — une probabilité, une distance cosinus, un
pourcentage sorti d'un LLM — donc elles sont fusionnées par leurs *rangs* (Reciprocal
Rank Fusion pondérée, cf. `fuse_candidates` et `SOURCE_WEIGHTS`) plutôt que par leurs
scores : un code que deux sources proposent chacune en 2e position passe devant un code
qu'une seule place en 1re, ce qui est exactement le signal qu'on veut donner à
l'annotateur. Le code déjà en place est retiré de la liste : la question posée est « si
le code est faux, quel serait le bon », le proposer serait du bruit.

Écrit `match_verifier_suggestions.parquet` à côté du parquet du run, en format long —
une ligne par (row_id, rang), colonnes `code`, `name`, `sources`, `fusion_score`,
`supervised_proba`. Un format long plutôt que des listes imbriquées parce qu'il se relit
tel quel en SQL (duckdb) pour mesurer après coup ce que ce suggesteur valait :
`sources` dit quelle méthode a proposé le code que l'annotateur a fini par retenir.

Le fichier est **facultatif** côté app : un run sans suggestions s'annote comme avant,
avec le seul champ libre. Le script refuse en revanche d'écraser un fichier existant
sans `--overwrite`, parce que changer les suggestions au milieu d'une campagne change ce
que les annotateurs ont vu sans que rien dans le log ne le dise.

Ce script ne mesure pas le prompt du MatchVerifier : pas de garde `assert_clean` ici (cf.
src.utils.run_provenance), le run visé est désigné par `--commit`/`--model` comme dans
l'app, pas déduit de HEAD au sens strict.

Nécessite à l'exécution : Neo4j et l'API d'embeddings (source `embedding`),
CODIF_APE_API_USERNAME/PASSWORD (source `supervised`). Une source dont
l'initialisation échoue est signalée et abandonnée, les autres continuent.

Usage :
    uv run -m src.evaluation.match_verifier_suggestions \
        --commit v0.0.1-1-g2441f54 --model qwen3-6-35b-moe
"""

import argparse
import asyncio
import logging
import os

import polars as pl

from src.evaluation.metrics import normalize_code
from src.evaluation.row_id import row_id_for
from src.utils import storage
from src.utils.logging import configure_logging
from src.utils.run_provenance import model_slug, revision_tag

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "s3://projet-ape/graal/data/eval/match_verifier_eval"
SUGGESTIONS_FILENAME = "match_verifier_suggestions.parquet"

# Colonnes lues dans le parquet du run et écrites dans celui-ci : dupliquées depuis
# match_verifier_eval.py / l'app plutôt qu'importées, même raison que dans l'app (ce
# module-là tire toute la pile agent à l'import).
DETAILS_FILENAME = "match_verifier_eval.parquet"
TEXT_COLUMN = "libelle"
CODE_COLUMN = "current_code"
ALTERNATIVE_CODE_COLUMN = "match_verifier_alternative_code"

DEFAULT_TOP_K = 5
ALL_SOURCES = ("supervised", "embedding", "alternative")

# Constante de la Reciprocal Rank Fusion. Elle amortit l'écart entre les premiers rangs :
# à k=60 (la valeur de l'article d'origine, et celle qu'utilisent les moteurs de
# recherche qui s'en servent), 1/(60+1) et 1/(60+2) sont presque égaux, si bien qu'un
# code trouvé par deux sources l'emporte sur un code que la seule source la plus fiable
# place en tête. C'est le comportement voulu : aucune des trois sources n'est assez bonne
# pour qu'on lui donne raison seule contre les deux autres.
RRF_K = 60

# Poids par source dans la fusion. Sans eux le classement serait décidé par le volume :
# la recherche vectorielle rend toujours ses `top_k` candidats, là où le modèle supervisé
# s'arrête de lui-même dès que la queue de distribution devient négligeable — un libellé
# sans ambiguïté ne lui tire qu'un seul écho. La source la moins sûre remplirait donc la
# liste à elle seule, et un voisin de notice arriverait à égalité avec la prédiction du
# modèle de production. Les poids rétablissent l'ordre voulu :
#
# - `supervised` : le modèle de production, la seule source dont le score soit une
#   probabilité calibrée, et la plus exacte prise isolément.
# - `alternative` : le concurrent que le MatchVerifier a nommé après avoir lu la notice
#   du code en place — un candidat raisonné, pas un voisin.
# - `embedding` : une similarité entre le libellé et des notices, dont
#   `evaluate_embeddings.py` existe précisément pour mesurer les limites. Comptée pour
#   moitié : assez pour départager, pas assez pour l'emporter seule sur les deux autres.
SOURCE_WEIGHTS = {"supervised": 1.0, "alternative": 1.0, "embedding": 0.5}

# Nombre d'appels en vol par source. Les deux sources appelantes sont des I/O réseau
# (HTTP pour le modèle supervisé, embeddings + Neo4j pour la recherche vectorielle) ;
# c'est le modèle supervisé, une API de production partagée, qui fixe la prudence ici.
DEFAULT_CONCURRENCY = 10


def fuse_candidates(
    ranked_by_source: dict[str, list[str]],
    exclude: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Fusionne des classements de sources hétérogènes par Reciprocal Rank Fusion pondérée.

    Chaque source donne une liste de codes ordonnée du meilleur au moins bon ; un code
    reçoit `poids / (rrf_k + rang)` de chaque source qui le cite, et les candidats sortent
    triés par la somme de ces contributions. Aucune des trois sources n'ayant d'échelle
    comparable aux autres (probabilité, distance, pourcentage LLM), c'est leur ordre —
    la seule chose qu'elles aient en commun — qui les fusionne, leur poids respectif
    disant le reste (cf. `SOURCE_WEIGHTS` ; une source non listée pèse 1).

    Les codes sont normalisés avant comparaison (cf. `normalize_code`), sans quoi les
    trois sources ne se rejoindraient jamais : le modèle supervisé rend « 1071H », la
    recherche vectorielle « 10.71H », et le MatchVerifier l'une ou l'autre forme selon
    les lignes. `exclude` (le code déjà en place) est retiré sous la même normalisation.

    À égalité de score, l'ordre alphabétique tranche : deux runs du script sur les mêmes
    entrées doivent proposer la même liste, dans le même ordre.
    """
    excluded = normalize_code(exclude)
    weights = SOURCE_WEIGHTS if weights is None else weights
    fused: dict[str, dict] = {}
    for source, codes in ranked_by_source.items():
        seen: set[str] = set()
        for rank, raw in enumerate(codes, start=1):
            code = normalize_code(raw)
            # Un doublon à l'intérieur d'une même source (les formes pointée et non
            # pointée du même code, par exemple) ne doit pas compter deux fois : ce
            # serait un vote double déguisé en accord entre sources.
            if code is None or code == excluded or code in seen:
                continue
            seen.add(code)
            entry = fused.setdefault(code, {"code": code, "sources": [], "fusion_score": 0.0})
            entry["sources"].append(source)
            entry["fusion_score"] += weights.get(source, 1.0) / (rrf_k + rank)
    ordered = sorted(fused.values(), key=lambda e: (-e["fusion_score"], e["code"]))
    return ordered[:top_k]


class CandidateSources:
    """Les trois sources de candidats, chacune facultative.

    Regroupées derrière un objet plutôt que passées en paramètres parce qu'elles
    partagent un cycle de vie : celles qui n'ont pas pu s'initialiser (Neo4j
    injoignable, identifiants d'API absents) sont désactivées une fois pour toutes au
    démarrage, et le reste du script n'a plus à savoir lesquelles tournent.
    """

    def __init__(self, sources: list[str], top_k: int):
        self.top_k = top_k
        self.graph = None
        self.supervised = None

        if "embedding" in sources:
            try:
                from src.config import neo4j_config
                from src.neo4j_graph.graph import Graph

                self.graph = Graph(neo4j_config)
            except Exception:
                logger.exception("Source 'embedding' unavailable (Neo4j/embeddings), skipping it")
        if "supervised" in sources:
            try:
                from src.agents.Text2Code.classifiers.supervised_classifier import (
                    SupervisedClassifier,
                )

                self.supervised = SupervisedClassifier()
            except Exception:
                logger.exception("Source 'supervised' unavailable (codif-ape-API), skipping it")

        self.use_alternative = "alternative" in sources
        enabled = [
            name
            for name, on in (
                ("supervised", self.supervised is not None),
                ("embedding", self.graph is not None),
                ("alternative", self.use_alternative),
            )
            if on
        ]
        if not enabled:
            raise SystemExit(
                f"None of the requested sources ({', '.join(sources)}) could be initialized; "
                "there would be nothing to suggest."
            )
        logger.info(f"Candidate sources enabled: {', '.join(enabled)}")

    async def candidates_for(self, libelle: str, alternative_code: str | None) -> tuple[dict, dict]:
        """`(classements par source, probabilités du modèle supervisé par code)`.

        Une source qui échoue sur *cette* ligne (timeout HTTP, embedding refusé) est
        journalisée et laissée de côté : mieux vaut proposer deux candidats sur trois
        sources qu'abandonner la ligne, qui retomberait sur le champ libre seul.
        """
        ranked: dict[str, list[str]] = {}
        probas: dict[str, float] = {}

        if self.supervised is not None:
            try:
                predictions = await self.supervised.top_k(libelle, k=self.top_k)
                ranked["supervised"] = [p["code"] for p in predictions]
                probas = {
                    normalize_code(p["code"]): p["proba"]
                    for p in predictions
                    if normalize_code(p["code"])
                }
            except Exception:
                logger.warning(f"Supervised model failed on {libelle!r}", exc_info=True)
        if self.graph is not None:
            try:
                ranked["embedding"] = await self.graph.get_closest_codes(libelle, top_k=self.top_k)
            except Exception:
                logger.warning(f"Embedding search failed on {libelle!r}", exc_info=True)
        if self.use_alternative and alternative_code:
            ranked["alternative"] = [alternative_code]
        return ranked, probas

    def name_of(self, code: str) -> str | None:
        """Intitulé officiel du code, lu dans Neo4j (mis en cache par `Graph`).

        `Graph.get_code_information` retrouve aussi bien la forme pointée que la forme
        non pointée (cf. `_with_dotted_retry`), ce qui est indispensable ici : les codes
        sortent de `fuse_candidates` normalisés, donc sans point, alors que les noeuds du
        graphe sont pointés.
        """
        if self.graph is None:
            return None
        try:
            return self.graph.get_code_information(code).get("name")
        except Exception:
            logger.warning(f"Could not read the name of {code}", exc_info=True)
            return None


async def suggest_rows(
    rows: list[dict], sources: CandidateSources, top_k: int, concurrency: int
) -> list[dict]:
    """Une entrée par (ligne du run, rang du candidat), prête à écrire en parquet.

    Les lignes tournent concurremment, `concurrency` bornant les appels en vol, comme
    dans match_verifier_eval.verify_rows. Pas de checkpoint ici, à la différence de
    l'éval : le calcul entier tient en quelques minutes et se relance à l'identique,
    donc une interruption ne coûte rien qu'il vaille la peine de reprendre.
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(rows)
    per_row: list[list[dict]] = [[] for _ in range(total)]

    async def handle(i: int, row: dict) -> None:
        libelle = row[TEXT_COLUMN]
        current_code = str(row[CODE_COLUMN])
        async with semaphore:
            ranked, probas = await sources.candidates_for(libelle, row.get(ALTERNATIVE_CODE_COLUMN))
        candidates = fuse_candidates(ranked, exclude=current_code, top_k=top_k)
        per_row[i] = [
            {
                "row_id": row_id_for(libelle, current_code),
                TEXT_COLUMN: libelle,
                CODE_COLUMN: current_code,
                "rank": rank,
                "code": candidate["code"],
                "name": sources.name_of(candidate["code"]),
                # Les sources qui ont proposé ce code, dans l'ordre où elles sont
                # interrogées : une chaîne plutôt qu'une liste, pour que le parquet reste
                # plat et se lise en SQL sans déballer de type imbriqué.
                "sources": ",".join(candidate["sources"]),
                "fusion_score": candidate["fusion_score"],
                "supervised_proba": probas.get(candidate["code"]),
            }
            for rank, candidate in enumerate(candidates, start=1)
        ]
        if (i + 1) % 100 == 0 or i + 1 == total:
            logger.info(f"{i + 1}/{total} rows done")

    await asyncio.gather(*(handle(i, row) for i, row in enumerate(rows)))
    return [entry for entries in per_row for entry in entries]


SUGGESTIONS_SCHEMA = [
    ("row_id", pl.Utf8),
    (TEXT_COLUMN, pl.Utf8),
    (CODE_COLUMN, pl.Utf8),
    ("rank", pl.Int64),
    ("code", pl.Utf8),
    ("name", pl.Utf8),
    ("sources", pl.Utf8),
    ("fusion_score", pl.Float64),
    ("supervised_proba", pl.Float64),
]


async def run(args) -> int:
    commit = args.commit or revision_tag()
    model = model_slug(args.model)
    run_dir = os.path.join(args.input, commit, model)
    details_path = os.path.join(run_dir, DETAILS_FILENAME)
    if not storage.path_exists(details_path):
        raise SystemExit(
            f"No eval run for {commit}/{model}: {details_path} does not exist. Produce one "
            "with `uv run -m src.evaluation.match_verifier_eval`, or point --commit/--model "
            "at an existing run."
        )
    output_path = os.path.join(run_dir, SUGGESTIONS_FILENAME)
    if storage.path_exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"{output_path} already exists. Recomputing it would change what reviewers are "
            "shown mid-campaign, without anything in the judgment log saying so — pass "
            "--overwrite if that is what you want."
        )

    with storage.open_path(details_path, "rb") as f:
        details = pl.read_parquet(f)
    logger.info(f"Suggesting up to {args.top_k} codes for {len(details)} rows of {commit}/{model}")

    sources = CandidateSources(args.sources, args.top_k)
    entries = await suggest_rows(details.to_dicts(), sources, args.top_k, args.concurrency)

    suggestions = pl.DataFrame(entries, schema=SUGGESTIONS_SCHEMA)
    storage.makedirs(run_dir)
    with storage.open_path(output_path, "wb") as f:
        suggestions.write_parquet(f)

    n_rows_covered = suggestions["row_id"].n_unique()
    by_source = {
        source: int(suggestions["sources"].str.contains(source).sum()) for source in ALL_SOURCES
    }
    logger.info(
        f"Wrote {len(suggestions)} suggestions for {n_rows_covered}/{len(details)} rows to "
        f"{output_path}; candidates per source: {by_source}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute the candidate codes shown to reviewers by the MatchVerifier "
        "review app"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Root directory of the eval runs (default: {DEFAULT_INPUT}); the suggestions "
        f"are written next to the run's {DETAILS_FILENAME}",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Which eval run to suggest for, by the commit tag naming its directory "
        "(default: the current HEAD's tag)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Which eval run to suggest for, by the model naming its directory "
        "(default: GENERATION_MODEL)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Candidates kept per row after fusion (default: {DEFAULT_TOP_K}); also how "
        "many each source is asked for",
    )
    parser.add_argument(
        "--sources",
        default=",".join(ALL_SOURCES),
        help=f"Comma-separated candidate sources among {', '.join(ALL_SOURCES)} "
        "(default: all three)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max rows in flight at once (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing suggestions file for this run",
    )
    args = parser.parse_args()

    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(args.sources) - set(ALL_SOURCES)
    if unknown:
        parser.error(f"Unknown source(s): {', '.join(sorted(unknown))}")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
