"""Cohérence de l'annotateur et de Sirus sur un même libellé (vague 1 Budget de famille).

Pour chaque libellé présent au moins deux fois dans le fichier source, on regarde si
l'annotateur donne toujours le même code (niveau 4, `code_lvl4`) et si Sirus donne
toujours le même code (`sirus_code`), puis on croise les deux.

Deux définitions du « même libellé » : texte brut identique (`raw_product`), et texte
normalisé (minuscules, espaces et ponctuation réduits). Et une variante où le libellé
est pris avec le type de magasin, qui fait partie de ce que voit Sirus.

Sortie : agrégats uniquement (aucun libellé), imprimés et écrits en JSON.

Usage :
    uv run python slides_coicop/sources/coherence_libelles.py \
        data/coicop/predictions_2026-09-08_codif-c9vjm.parquet <out.json>
"""
import json
import re
import sys

import pandas as pd

predictions, out = sys.argv[1:3]
d = pd.read_parquet(predictions)


def normalize(text):
    text = re.sub(r"[^\w]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


d["libelle_norm"] = d.raw_product.map(normalize)
d["shop_type_name"] = d.shop_type_name.fillna("<inconnu>")


def study(keys, name):
    g = d.groupby(keys)
    per = pd.DataFrame({
        "n": g.size(),
        "human_codes": g.code_lvl4.nunique(dropna=False),
        "sirus_codes": g.sirus_code.nunique(dropna=False),
        # part des lignes qui portent le code le plus fréquent du libellé
        "human_modal": g.code_lvl4.agg(lambda s: s.value_counts(dropna=False).iloc[0]),
        "sirus_modal": g.sirus_code.agg(lambda s: s.value_counts(dropna=False).iloc[0]),
    })
    rep = per[per.n >= 2].copy()
    rep["humain"] = rep.human_codes.eq(1).map({True: "toujours le même code", False: "codes différents"})
    rep["sirus"] = rep.sirus_codes.eq(1).map({True: "toujours le même code", False: "codes différents"})
    by_label = pd.crosstab(rep.humain, rep.sirus, margins=True, margins_name="Total")
    by_rows = pd.crosstab(rep.humain, rep.sirus, values=rep.n, aggfunc="sum", margins=True,
                          margins_name="Total").fillna(0).astype(int)
    res = {
        "definition": name,
        "n_libelles_total": int(len(per)),
        "n_libelles_repetes": int(len(rep)),
        "n_lignes_repetees": int(rep.n.sum()),
        "humain_toujours_meme_code": float(rep.human_codes.eq(1).mean()),
        "sirus_toujours_meme_code": float(rep.sirus_codes.eq(1).mean()),
        "humain_part_lignes_code_majoritaire": float(rep.human_modal.sum() / rep.n.sum()),
        "sirus_part_lignes_code_majoritaire": float(rep.sirus_modal.sum() / rep.n.sum()),
        "matrice_libelles": by_label.to_dict(),
        "matrice_lignes": by_rows.to_dict(),
    }
    print(f"\n=== {name} ===")
    print({k: v for k, v in res.items() if not k.startswith("matrice")})
    print("Libellés :\n", by_label.to_string())
    print("Lignes :\n", by_rows.to_string())
    return res


results = [
    study("raw_product", "libellé brut identique"),
    study("libelle_norm", "libellé normalisé"),
    study(["raw_product", "shop_type_name"], "libellé brut + type de magasin"),
]
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
