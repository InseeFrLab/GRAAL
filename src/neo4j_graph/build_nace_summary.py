"""Construit un résumé textuel de la hiérarchie NACE à partir de Neo4j (par défaut,
niveau 5 : la nomenclature complète, jusqu'aux codes terminaux inclus — seulement code
et nom, pas la notice complète).

Ce résumé est le contexte d'orientation donné d'emblée à `SummaryAgenticClassifier`
(cf. `src/agents/Text2Code/classifiers/summary_classifier.py`) : plutôt que de partir
sans aucune vue d'ensemble (comme `NavigatorAgenticClassifier`, qui découvre la
hiérarchie noeud par noeud) ou d'un point de départ par similarité d'embedding (comme
`AgenticRAGClassifier`), ce classifieur reçoit la structure globale de la nomenclature
dans son prompt système et choisit librement quels codes approfondir via les outils
de `Graph`.

Nécessite à l'exécution la base Neo4j configurée dans l'environnement (mêmes
prérequis que src.main).

Usage:
    uv run -m src.neo4j_graph.build_nace_summary --max-level 5 --output data/nace_summary.txt
"""

import argparse
import logging

from src.config import neo4j_config
from src.neo4j_graph.graph import Graph
from src.utils import storage
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def build_summary_text(rows: list[dict], max_level: int) -> str:
    lines = [f"Résumé de la nomenclature NACE (niveaux 1 à {max_level}) :", ""]
    for row in rows:
        indent = "  " * (row["level"] - 1)
        lines.append(f"{indent}{row['code']} - {row['name']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a textual NACE summary from Neo4j")
    parser.add_argument(
        "--max-level",
        type=int,
        default=5,
        help="Deepest level to include (default: 5 = full NAF hierarchy down to terminal codes)",
    )
    parser.add_argument("--output", default="data/nace_summary.txt", help="Output text file path")
    args = parser.parse_args()

    graph = Graph(neo4j_config)
    rows = graph.get_summary_tree(args.max_level)
    text = build_summary_text(rows, args.max_level)

    with storage.open_path(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(
        f"Wrote NACE summary ({len(rows)} codes, levels<={args.max_level}) to {args.output}"
    )
    print(f"Wrote {len(rows)} codes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
