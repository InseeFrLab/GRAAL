"""Compare un run d'`eval_summary` au modèle supervisé de production (codif-ape-API).

Relit le `predictions.jsonl` d'un run et classe les mêmes libellés avec
`SupervisedClassifier`. Ajoute à chaque ligne `prod_pred` et `prod_confidence` (écrits
dans `predictions_prod.jsonl`), puis calcule dans `metrics_prod.json` :
    - l'exactitude par niveau et la vue « label » (par niveau d'accord, par source)
      pour trois bras : prod, agent seul (classifier) et agent + vérifieur ;
    - la table de contingence agent / prod à la sous-classe (les deux justes, agent
      seul, prod seule, aucun des deux) ;
    - l'exactitude de la prod selon sa confiance (IC), pour situer les cas où l'agent
      aurait le plus à apporter.

Les libellés sont exactement ceux du run, pour une comparaison appariée sans relancer
l'agent. Il faut CODIF_APE_API_USERNAME / CODIF_APE_API_PASSWORD dans `.env`.

Usage :
    uv run python -m eval.prod_baseline eval/results/n300_v2
"""

import argparse
import asyncio
import json
import os

from eval.eval_summary import accuracy, is_correct, label_metrics  # charge aussi .env
from src.agents.Text2Code.classifiers.supervised_classifier import SupervisedClassifier


async def predict_all(texts: list[str], concurrency: int) -> list[tuple[str | None, float | None]]:
    clf = SupervisedClassifier()
    semaphore = asyncio.Semaphore(concurrency)

    async def one(text):
        async with semaphore:
            try:
                out = await clf(text)
                return out.code, out.proposed_confidence
            except Exception as e:  # une erreur d'API ne doit pas faire perdre le run
                print(f"prod KO pour {text[:50]!r}: {e!r}")
                return None, None

    return await asyncio.gather(*(one(t) for t in texts))


def contingency(rows: list[dict], agent_key: str) -> dict:
    agent = [is_correct(r.get(agent_key), r["gold"], None) for r in rows]
    prod = [is_correct(r.get("prod_pred"), r["gold"], None) for r in rows]
    pairs = list(zip(agent, prod))
    return {
        "both_right": pairs.count((True, True)),
        "agent_only": pairs.count((True, False)),
        "prod_only": pairs.count((False, True)),
        "both_wrong": pairs.count((False, False)),
        "same_prediction": sum(
            is_correct(r.get(agent_key), r.get("prod_pred"), None) for r in rows
        ),
    }


def by_prod_confidence(rows: list[dict], agent_key: str, thresholds=(0.5, 0.8)) -> dict:
    """Exactitude prod et agent par tranche de confiance de la prod."""
    edges = [0.0, *thresholds, 1.01]
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        sub = [
            r
            for r in rows
            if r.get("prod_confidence") is not None and lo <= r["prod_confidence"] < hi
        ]
        if not sub:
            continue
        out[f"[{lo},{min(hi, 1.0)}]"] = {
            "n": len(sub),
            "prod": round(
                sum(is_correct(r["prod_pred"], r["gold"], None) for r in sub) / len(sub), 3
            ),
            "agent": round(
                sum(is_correct(r.get(agent_key), r["gold"], None) for r in sub) / len(sub), 3
            ),
        }
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="dossier d'un run eval_summary (predictions.jsonl)")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "predictions.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    preds = await predict_all([r["text"] for r in rows], args.concurrency)
    for r, (code, conf) in zip(rows, preds):
        r["prod_pred"], r["prod_confidence"] = code, conf

    with open(os.path.join(args.run_dir, "predictions_prod.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arms = {"prod": "prod_pred", "classifier": "pred"}
    if any("pred_verified" in r for r in rows):
        arms["classifier+verifier"] = "pred_verified"
    metrics = {
        "n": len(rows),
        "prod_errors": sum(r["prod_pred"] is None for r in rows),
        "accuracy": {arm: accuracy(rows, key) for arm, key in arms.items()},
        "label_view": {arm: label_metrics(rows, key) for arm, key in arms.items()},
        "contingency_vs_prod": {
            arm: contingency(rows, key) for arm, key in arms.items() if arm != "prod"
        },
        "by_prod_confidence": {
            arm: by_prod_confidence(rows, key) for arm, key in arms.items() if arm != "prod"
        },
    }
    with open(os.path.join(args.run_dir, "metrics_prod.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
