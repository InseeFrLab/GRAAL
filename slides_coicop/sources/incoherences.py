"""Variations d'un même libellé : contexte, codes suspects, ancrage de l'annotateur.

1. Le contexte (type de magasin, magasin, achat en ligne) explique-t-il qu'un même
   libellé reçoive plusieurs codes, chez l'annotateur et chez Sirus ?
2. Codes « suspects » : désaccord de division entre l'annotateur et Sirus, tranché par
   deux avis indépendants de l'un et de l'autre (vérificateur, second modèle).
3. Ancrage : l'annotateur reprend-il la prédiction automatique qu'on lui montrait ?

Imprime des agrégats ; avec --examples, imprime aussi des exemples (libellés : usage
interne uniquement, secret statistique).

Usage :
    uv run python slides_coicop/sources/incoherences.py <predictions.parquet> <run_dir> \
        <coicop_summary.txt> <out.json> [--examples]
"""
import json
import re
import sys

import pandas as pd

predictions, run_dir, summary_txt, out = sys.argv[1:5]
show = "--examples" in sys.argv

names = {}
for line in open(summary_txt, encoding="utf-8"):
    m = re.match(r"\s*([0-9][0-9.]*) - (.*)", line)
    if m:
        names[m.group(1)] = m.group(2).strip()


def name(code):
    return f"{code} {names.get(str(code), '?')[:45]}"


d = pd.read_parquet(predictions)
d["h"] = d.code_lvl4.astype(str)
d["s"] = d.sirus_code.astype(str)
d["shop_type"] = d.shop_type_name.fillna("<inconnu>")
d["shop_n"] = d["shop"].fillna("<inconnu>").str.lower().str.strip()
res = {}

# ---------------------------------------------------------------- 1. contexte
rep = d.groupby("raw_product").filter(lambda t: len(t) >= 2)


def explained(col_code, ctx):
    """Parmi les libellés à codes multiples, part qui deviennent constants une fois le
    contexte fixé (chaque groupe libellé × contexte n'a plus qu'un code)."""
    var = rep.groupby("raw_product")[col_code].nunique()
    var = var[var > 1].index
    sub = rep[rep.raw_product.isin(var)]
    per = sub.groupby(["raw_product"] + ctx)[col_code].nunique().groupby("raw_product").max()
    return int(len(var)), float((per == 1).mean())


ctx_sets = {
    "type de magasin": ["shop_type"],
    "magasin": ["shop_n"],
    "magasin + en ligne": ["shop_n", "INTERNET"],
}
res["contexte"] = {}
for who, col in [("annotateur", "h"), ("sirus", "s")]:
    res["contexte"][who] = {}
    for label, ctx in ctx_sets.items():
        n, share = explained(col, ctx)
        res["contexte"][who][label] = {"libelles_variables": n, "part_expliquee": share}
print("1. Part des libellés à codes multiples rendus constants par le contexte")
print(json.dumps(res["contexte"], ensure_ascii=False, indent=1))

# Même libellé, même magasin, codes humains de divisions différentes
same_ctx = rep.groupby(["raw_product", "shop_n"]).filter(lambda t: len(t) >= 2)
g = same_ctx.groupby(["raw_product", "shop_n"])
div_conflict = g.h.agg(lambda s: s.str[:2].nunique() > 1)
res["meme_magasin_division_differente_annotateur"] = int(div_conflict.sum())
res["meme_magasin_groupes"] = int(len(div_conflict))
div_conflict_s = g.s.agg(lambda s: s.str[:2].nunique() > 1)
res["meme_magasin_division_differente_sirus"] = int(div_conflict_s.sum())
print(f"\nMême libellé ET même magasin ({len(div_conflict)} groupes) : division différente "
      f"chez l'annotateur {div_conflict.sum()}, chez Sirus {div_conflict_s.sum()}")

# ---------------------------------------------------------------- 2. suspects
v = pd.read_parquet(f"{run_dir}/match_verifier_eval.parquet")
sm = pd.read_parquet(f"{run_dir}/summary_classifier_predictions.parquet")
v = v.merge(sm[["libelle", "summary_code"]], on="libelle", how="left")
v = v[v.summary_code.notna()].copy()
v["s"], v["h"] = v.current_code.astype(str), v.expected_code.astype(str)
v["alt"] = v.match_verifier_alternative_code.astype(str)
v["sum4"] = [".".join(str(c).split(".")[:4]) for c in v.summary_code]
v = v[~v.h.str.startswith(("98", "99"))]
div = lambda c: c.str[:2]
conflict = v[div(v.s) != div(v.h)].copy()
# Annotateur suspect : le vérificateur valide nettement Sirus et le second modèle,
# qui ne voit pas le code Sirus, tombe dans la même division que Sirus.
hs = conflict[conflict.match_verifier_verdict & (conflict.match_verifier_match_score >= 80)
              & (div(conflict.sum4) == div(conflict.s))]
# Sirus suspect : le vérificateur le juge incompatible, son alternative et le second
# modèle tombent tous deux dans la division de l'annotateur.
ss = conflict[(conflict.match_verifier_match_score < 40)
              & (div(conflict.alt) == div(conflict.h)) & (div(conflict.sum4) == div(conflict.h))]
res["desaccords_division"] = {
    "paires": int(len(conflict)), "lignes": int(conflict.n_rows.sum()),
    "annotateur_suspect_paires": int(len(hs)), "annotateur_suspect_lignes": int(hs.n_rows.sum()),
    "sirus_suspect_paires": int(len(ss)), "sirus_suspect_lignes": int(ss.n_rows.sum()),
    "non_tranches_paires": int(len(conflict) - len(hs) - len(ss)),
}
print("\n2. Désaccords de division annotateur / Sirus (hors 98/99, paires classées par le 2e modèle)")
print(json.dumps(res["desaccords_division"], ensure_ascii=False, indent=1))
for label, x in [("annotateur suspect", hs), ("Sirus suspect", ss)]:
    top = (x.assign(paire=x.h.map(name) + "  <->  Sirus " + x.s.map(name))
           .groupby("paire").n_rows.sum().sort_values(ascending=False).head(8))
    print(f"\n   Confusions les plus fréquentes, {label} (lignes) :")
    print(top.to_string())
    res[f"top_{label.replace(' ', '_')}"] = top.to_dict()

# ---------------------------------------------------------------- 3. ancrage
prev = pd.Series([".".join(str(c).split(".")[:4]) for c in d.predicted_code_previous], index=d.index)
same_prev = prev == d.h
res["ancrage"] = {
    "annotateur_egal_prediction_montree": float(same_prev.mean()),
    "par_source": d.assign(x=same_prev).groupby("prediction_source_previous").x.mean().to_dict(),
    "part_source": d.prediction_source_previous.value_counts(normalize=True).to_dict(),
    "sirus_egal_prediction_montree": float((prev == d.s).mean()),
}
print("\n3. Ancrage")
print(json.dumps(res["ancrage"], ensure_ascii=False, indent=1))

with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)

if show:
    cols = ["libelle", "n_rows", "h", "s", "sum4", "match_verifier_match_score", "alt"]
    for label, x in [("ANNOTATEUR SUSPECT", hs), ("SIRUS SUSPECT", ss)]:
        print(f"\n--- Exemples {label} (usage interne) ---")
        print(x.sort_values("n_rows", ascending=False)[cols].head(12).to_string(index=False))
    print("\n--- Même libellé, même magasin, divisions différentes chez l'annotateur (usage interne) ---")
    keys = div_conflict[div_conflict].index[:10]
    for lib, shop in keys:
        t = same_ctx[(same_ctx.raw_product == lib) & (same_ctx.shop_n == shop)]
        print(f"{lib!r} @ {shop!r}: annotateur {t.h.value_counts().to_dict()} | Sirus {t.s.value_counts().to_dict()}")
