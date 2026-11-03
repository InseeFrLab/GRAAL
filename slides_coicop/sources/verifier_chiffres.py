"""Recalcule depuis le parquet du run les chiffres cités dans le texte d'analyse."""
import sys

import pandas as pd

d = pd.read_parquet(sys.argv[1])
print(d[["current_code", "expected_code", "match_verifier_alternative_code"]].head(3).to_string())


def norm(c):
    return None if c is None else str(c).replace(".", "").strip()


d["s"] = d.current_code.map(norm)
d["h"] = d.expected_code.map(norm)
d["a"] = d.match_verifier_alternative_code.map(norm)
d["rej"] = ~d.match_verifier_verdict
err = ~d.expected_match_branch

print("pairs", len(d), "errors(branch)", err.sum())
print("recall", d.rej[err].mean(), "false_rej", d.rej[~err].mean())

# pondéré par lignes
w = d.n_rows
print("row-weighted recall", (d.rej & err).mul(w).sum() / w[err].sum(),
      "row-weighted false_rej", (d.rej & ~err).mul(w).sum() / w[~err].sum())

# par division (code Sirus)
d["div"] = d.s.str[:2]
d["hdiv"] = d.h.str[:2]
notech = ~d.h.str.startswith(("98", "99"))
for col, div, mask in [(c, v, m) for c in ["div", "hdiv"] for v in ["11", "06"] for m in ["all", "notech"]]:
    x = d[(d[col] == div) & (notech if mask == "notech" else True)]
    print(col, mask, end=" ")
    e = ~x.expected_match_branch
    print(f"div {div}: n={len(x)} errors={e.sum()} recall={x.rej[e].mean():.3f} false_rej={x.rej[~e].mean():.3f}")

# alternative = code jugé / sous-code
print("alt == judged", (d.a == d.s).mean())
print("alt strict subcode of judged", pd.Series([a != b and a.startswith(b) for a, b in zip(d.a.fillna(""), d.s.fillna(""))]).mean())

# incohérence verdict / scores
inc = d.match_verifier_verdict & (d.match_verifier_alternative_score - d.match_verifier_match_score >= 20)
print("accept despite alt >= +20", inc.sum(), inc.mean())

# p_match quasi binaire
p = d.match_verifier_p_match
print("p<0.1", (p < 0.1).mean(), "p>0.9", (p > 0.9).mean())
print("mean duration", d.match_verifier_duration_s.mean(), "total h", d.match_verifier_duration_s.sum() / 3600)
print("rows covered", w.sum())
