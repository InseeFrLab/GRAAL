"""Liste les lignes écartées de l'évaluation MatchVerifier COICOP.

match_verifier_eval écarte les lignes dont le code Sirus est absent du graphe
(`n_rows_unknown_code` = 400). Le graphe n'étant pas joignable hors Datalab, on
reconstitue la liste à partir du résumé de la nomenclature (`data/coicop_summary.txt`),
généré depuis ce même graphe : un code y figure si et seulement s'il est dans le graphe.

Sorties (contiennent des libellés : secret statistique, ne pas versionner) :
- <out>/lignes_ecartees.csv : une ligne par ligne écartée ;
- <out>/lignes_ecartees_par_code.csv : agrégat par code Sirus, sans libellé.

Usage :
    uv run python slides_coicop/sources/lignes_ecartees.py \
        data/coicop/predictions_2026-09-08_codif-c9vjm.parquet data/coicop_summary.txt <out>
"""
import os
import re
import sys

import pandas as pd

predictions, summary, out = sys.argv[1:4]

pattern = re.compile(r"\s*([0-9][0-9.]*) - ")
with open(summary, encoding="utf-8") as f:
    graph_codes = {m.group(1) for line in f if (m := pattern.match(line))}

d = pd.read_parquet(predictions)
x = d[~d.sirus_code.isin(graph_codes)].copy()
assert len(x) == 400, f"{len(x)} lignes écartées, 400 attendues (n_rows_unknown_code)"

human = x.code_lvl4.fillna("")
x["humain_code_technique"] = human.str.match(r"^(98|99)")
x["accord_exact"] = human == x.sirus_code
x["meme_branche"] = [h == s or h.startswith(s + ".") for h, s in zip(human, x.sirus_code)]

os.makedirs(out, exist_ok=True)
cols = ["id", "raw_product", "sirus_code", "code", "code_lvl4",
        "humain_code_technique", "accord_exact", "meme_branche"]
x.sort_values(["sirus_code", "raw_product"])[cols].to_csv(
    os.path.join(out, "lignes_ecartees.csv"), index=False, sep=";", encoding="utf-8-sig")

agg = (x.groupby("sirus_code")
        .agg(lignes=("id", "size"), libelles_distincts=("raw_product", "nunique"),
             humain_aussi_98_99=("humain_code_technique", "sum"),
             accord_exact=("accord_exact", "sum"), meme_branche=("meme_branche", "sum"))
        .sort_values("lignes", ascending=False))
agg.loc["Total"] = [len(x), x.raw_product.nunique(), x.humain_code_technique.sum(),
                    x.accord_exact.sum(), x.meme_branche.sum()]
agg.to_csv(os.path.join(out, "lignes_ecartees_par_code.csv"), sep=";", encoding="utf-8-sig")
print(agg.to_string())
