# GRAAL

### Graph-based Reasoning Agents for Automatic Labelling

GRAAL is a research-oriented toolkit that combines graph vectorial database, knowledge reasoning, and modular agent components to support agentic automated data labelling in hierarchical classification, a common use case for National Statistical Institutes (NSIs).

The hierarchy of the nomenclature is represented as a graph database (Neo4j), enabling efficient traversal and reasoning over classification codes and their relationships.

It is intended to be used with reasoning LLM agents (that support tool-calling).

## Key features

- Graph (Neo4j) for representing the classification codes and relations
- Neo4j-backed tools for agentic workflows
- Utilities for building, embedding and managing graph data
- Modular classifiers and navigators to support multi-stage reasoning pipelines

## Quick start

1. Clone the repository:

	git clone https://github.com/InseeFrLab/GRAAL.git
	cd GRAAL

2. Install [ ``uv``](https://docs.astral.sh/uv/) (recommended) - via ``pip install uv`` for instance. Create a virtual environment and install dependencies :

	uv sync

3. Run a small experiment:

    uv run -m src.test

## Repository layout

Important folders and files:

- `src/` — main Python package
  - `agents/` — agent implementations and subcomponents (Code2Text, Text2Code, closers)
  - `neo4j_graph/` — graph building and helpers for Neo4j-backed graphs
  - `navigator/` — navigator logic: travel from the root to leaves in the graph, explaining each step
  - `utils/` — utility modules (logging, parser)
- `presentation/` — presentation materials and templates
- `pyproject.toml` — project metadata and dependencies

## Architecture overview

At a high level, GRAAL composes three concerns:

1. Knowledge Graph: A graph database (Neo4j or in-memory representation) stores facts, entities and provenance. The `neo4j_graph` package provides builders and helpers to construct and query this graph.
2. Agents: Reusable agent building blocks implement specific capabilities. For instance, `Code2Text` takes a label (code) as input and generates synthetic texts; `Text2Code` assists in classifying textual specifications in the given classification.
3. Connectors & Utilities: Embedding helpers, DB managers and parsers make it easy to populate the graph and wire agents into pipelines.

This modular separation allows mixing and matching pieces for experiments or production prototypes.

## Contributing

Contributions are welcome. A few guidelines:

1. Open an issue to discuss significant changes before implementing them.
2. Follow the existing code style and add tests for new behavior.
3. Keep changes focused and create feature branches for PRs.

If you add or modify graph schema, please include migration steps or a small script to populate example data.

## Roadmap & ideas

- Add integration tests for agent pipelines and graph persistence
- Implement more example notebooks demonstrating common workflows

## License

This project includes a `LICENSE` file in the repository root. Please refer to it for licensing details.

## Authors & contact

Maintained by InseeFrLab. For questions or contributions, please open an issue or contact the maintainers via the project repository.

