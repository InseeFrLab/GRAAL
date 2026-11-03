"""Désaccords « restauration (11) vs produit (01/02) » : le magasin donne-t-il raison à l'annotateur ?

Le vérificateur et le second modèle ne voient que le libellé ; l'annotateur voit aussi le
magasin. On regarde, pour les paires où annotateur et Sirus se séparent entre la
division 11 (restauration) et les divisions 01/02 (produits alimentaires, boissons),
dans quel type de magasin l'achat a eu lieu.

Usage :
    uv run python slides_coicop/sources/contexte_restauration.py <predictions.parquet> <run_dir> <out.json>
"""
import json
import re
import sys

import pandas as pd

predictions, run_dir, out = sys.argv[1:4]
d = pd.read_parquet(predictions)
v = pd.read_parquet(f"{run_dir}/match_verifier_eval.parquet")

RESTO_TYPES = re.compile(r"^(Bars|Restauration)", re.I)
RESTO_NAMES = re.compile(r"\b(caf[eé]|bar|brasserie|restau|pizzeria|bistro|snack|kebab|"
                         r"mcdo|burger|sushi|cr[eê]perie|traiteur)\b", re.I)


def context(row):
    t, s = row.shop_type_name, str(row["shop"] or "")
    if isinstance(t, str):
        return "type : restauration / bar" if RESTO_TYPES.match(t) else "type : autre commerce"
    if RESTO_NAMES.search(s):
        return "type inconnu, nom évoque un café/restaurant"
    return "type inconnu, autre nom"


d["contexte"] = d.apply(context, axis=1)
h, s = d.code_lvl4.astype(str), d.sirus_code.astype(str)
food = ("01", "02")
cases = {
    "annotateur 11, Sirus 01/02": h.str.startswith("11") & s.str[:2].isin(food),
    "annotateur 01/02, Sirus 11": h.str[:2].isin(food) & s.str.startswith("11"),
    "tous deux 11": h.str.startswith("11") & s.str.startswith("11"),
    "tous deux 01/02": h.str[:2].isin(food) & s.str[:2].isin(food),
}
res = {}
for label, m in cases.items():
    res[label] = {"lignes": int(m.sum()),
                  **d[m].contexte.value_counts(normalize=True).round(3).to_dict()}
print(pd.DataFrame(res).T.fillna(0).to_string())

# Ce que le vérificateur a fait de ces paires (il ne voit pas le magasin)
v["h"], v["s"] = v.expected_code.astype(str), v.current_code.astype(str)
m = v.h.str.startswith("11") & v.s.str[:2].isin(food)
res["verificateur_sur_annotateur11_sirus01"] = {
    "paires": int(m.sum()),
    "rejet_de_sirus": float((~v[m].match_verifier_verdict).mean()),
}
m2 = v.h.str[:2].isin(food) & v.s.str.startswith("11")
res["verificateur_sur_annotateur01_sirus11"] = {
    "paires": int(m2.sum()),
    "rejet_de_sirus": float((~v[m2].match_verifier_verdict).mean()),
}
print(json.dumps({k: res[k] for k in list(res)[-2:]}, ensure_ascii=False, indent=1))
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
