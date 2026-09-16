# Image de l'app de revue humaine du MatchVerifier
# (src/evaluation/apps/match_verifier_eval_app.py), déployée sur le SSP Cloud pour
# que les trois annotateurs travaillent sur une même instance plutôt que chacun sur
# un Flask local.
#
# Les dépendances viennent de uv.lock (`--frozen`), soit exactement ce qu'un `uv sync`
# donne en local : l'app n'a besoin que de flask/polars/s3fs/neo4j, mais
# src.neo4j_graph.graph — importé pour afficher les notices — tire toute la pile
# langchain + openai-agents, et résoudre un sous-ensemble choisi à la main en marge du
# lock est une dérive programmée.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/app \
    PATH="/app/.venv/bin:$PATH"

# Utilisateur non root : les clusters Onyxia refusent les pods qui tournent en root.
# Le `|| true` couvre le cas où l'uid 1000 existe déjà dans une future révision de
# l'image de base ; le mkdir/chown qui suit garantit dans tous les cas un HOME
# inscriptible, où certaines bibliothèques posent leur cache.
RUN (useradd --uid 1000 --create-home --home-dir /home/app --shell /bin/bash app || true) \
 && mkdir -p /home/app && chown 1000:1000 /home/app

WORKDIR /app

# Couche dépendances d'abord : c'est de loin la plus chère, et seuls pyproject.toml /
# uv.lock l'invalident — une modification de code seul se reconstruit en secondes.
# Pas de [build-system] dans pyproject.toml, donc uv n'installe que les dépendances :
# le code est importé depuis /app, présent sur sys.path via `python -m`.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src

USER 1000
EXPOSE 5052

ENTRYPOINT ["python", "-m", "src.evaluation.apps.match_verifier_eval_app"]
# Aucun défaut de service : l'image n'embarque pas .git, donc --commit et --model
# (qui valent sinon le tag de HEAD et GENERATION_MODEL) doivent être passés
# explicitement, comme le fait le Deployment. Sans arguments, l'image documente
# les siens.
CMD ["--help"]
