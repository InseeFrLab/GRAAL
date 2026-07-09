"""
Diagnostic de l'espace d'embedding : notices (NAF2025) vs libellés (exemples).

Le classifieur Agentic RAG (src/agents/Text2Code/classifiers/agentic_rag.py) se sert
d'une recherche par similarité entre le libellé d'activité à classer et les embeddings
des notices de nomenclature stockées dans Neo4j (src/neo4j_graph/graph.py:get_closest_codes)
pour choisir un code de départ. Ce script évalue la qualité de cette recherche
indépendamment de la navigation LLM qui la suit, sur un petit échantillon annoté :

  - quantitatif : accuracy@1 et recall@k du k-NN libellé -> notice (cosinus)
  - visuel : projection 2D (UMAP / PaCMAP / t-SNE / PCA) des notices et des libellés,
    avec les arêtes k-NN correctes/incorrectes et la vérité terrain

et permet de comparer plusieurs modèles d'embedding candidats (déployés derrière le
même endpoint URL_EMBEDDING_API) en éditant CANDIDATE_MODELS ci-dessous.

Usage:
    python evaluate_embeddings.py
"""

# %% Imports
import json
import os

import numpy as np
import pacmap
import pandas as pd
import plotly.graph_objects as go
import umap
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from src.neo4j_graph.graph_builder.config import COLUMNS_TO_KEEP, MAX_TOKENS, URL_EMBEDDING_API
from src.neo4j_graph.graph_builder.utils.embed_manager import truncate_docs_to_max_tokens
from src.neo4j_graph.graph_builder.utils.notice_manager import load_notices

# %% Config
# Add more model names deployed behind URL_EMBEDDING_API to compare them.
CANDIDATE_MODELS = [os.environ["EMBEDDING_MODEL"]]

EVAL_SAMPLE_PATH = "data/eval/eval_set_sample30.parquet"
# Ground truth (apet2025) is NAF2025-coded, so notices must come from the NAF2025
# nomenclature regardless of whatever NOTICES_PATH graph_builder/config is set to.
NAF2025_NOTICES_PATH = "projet-ape/notices/Notices-NAF2025-FR.parquet"

K_NN = 5
REDUCTION_METHODS = ["umap", "pacmap", "tsne", "pca"]
OUTPUT_DIR = "data/eval/embedding_diagnostics"


def normalize_code(code: str) -> str:
    return str(code).replace(".", "").replace(" ", "").upper()


def build_embedding_model(model_name: str) -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False is required for self-hosted, non-OpenAI models:
    # otherwise langchain pre-tokenizes the text with tiktoken (falling back to
    # cl100k_base for unrecognized model names) and sends the resulting integer
    # token IDs as `input`, which the server then decodes with its own tokenizer's
    # vocabulary — silently corrupting every embedding. Disabling it sends plain text.
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=URL_EMBEDDING_API,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        check_embedding_ctx_length=False,
    )


# %% Data loading
def load_eval_sample(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)[["libelle", "apet2025"]].dropna()
    df["apet2025"] = df["apet2025"].map(normalize_code)
    return df.reset_index(drop=True)


def load_leaf_notices(path: str) -> pd.DataFrame:
    df = load_notices(path, COLUMNS_TO_KEEP)
    df = df[df["FINAL"] == 1].copy()
    df["CODE"] = df["CODE"].map(normalize_code)
    df["text_to_embed"] = (
        df["NAME"].fillna("")
        + "\n"
        + df["Implementation_rule"].fillna("")
        + "\n"
        + df["Includes"].fillna("")
        + "\n"
        + df["IncludesAlso"].fillna("")
    )
    return df.reset_index(drop=True)


def truncate_texts(texts: list[str], max_tokens: int) -> list[str]:
    docs = [Document(page_content=t) for t in texts]
    docs = truncate_docs_to_max_tokens(docs, max_tokens)
    return [d.page_content for d in docs]


# %% k-NN retrieval
def compute_knn(label_embeddings: np.ndarray, notice_embeddings: np.ndarray, k: int):
    """For each libellé embedding, find its k nearest notice embeddings (cosine)."""
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(notice_embeddings)
    distances, indices = nbrs.kneighbors(label_embeddings)
    return indices, distances


def score_retrieval(
    indices: np.ndarray, distances: np.ndarray, notice_codes: list[str], target_codes: list[str]
):
    n = len(target_codes)
    hits_at_1 = 0
    hits_at_k = 0
    correct_sims, incorrect_sims = [], []
    edges = []  # (label_idx, notice_idx, distance, is_correct) — notice_idx indexes notice_codes

    for label_idx, (idx_row, dist_row, target) in enumerate(zip(indices, distances, target_codes)):
        retrieved_codes = [notice_codes[j] for j in idx_row]
        hits_at_1 += int(retrieved_codes[0] == target)
        hits_at_k += int(target in retrieved_codes)

        for notice_idx, dist in zip(idx_row, dist_row):
            is_correct = notice_codes[notice_idx] == target
            edges.append((label_idx, int(notice_idx), float(dist), is_correct))
            (correct_sims if is_correct else incorrect_sims).append(1 - dist)

    metrics = {
        "n": n,
        "accuracy_at_1": hits_at_1 / n,
        f"recall_at_{indices.shape[1]}": hits_at_k / n,
        "mean_cosine_sim_correct": float(np.mean(correct_sims)) if correct_sims else None,
        "mean_cosine_sim_incorrect": float(np.mean(incorrect_sims)) if incorrect_sims else None,
    }
    return metrics, edges


# %% Dimensionality reduction
def reduce_dimensions(embeddings: np.ndarray, method: str, random_state: int = 42):
    if method == "umap":
        reducer = umap.UMAP(
            random_state=random_state, n_neighbors=10, min_dist=0.1, metric="cosine"
        )
        return reducer.fit_transform(embeddings), "UMAP (cosine)"
    if method == "pacmap":
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            distance="angular",
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings), "PaCMAP (angular)"
    if method == "tsne":
        distances = cosine_distances(embeddings)
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, len(embeddings) - 1),
            metric="precomputed",
            init="random",
        )
        return reducer.fit_transform(distances), "t-SNE (cosine)"
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        return coords, f"PCA (variance: {reducer.explained_variance_ratio_.sum():.1%})"
    raise ValueError(f"Unknown reduction method: {method}")


# %% Plotting
def build_comparison_figure(
    combined_embeddings: np.ndarray,
    n_notices: int,
    notice_hover: list[str],
    label_hover: list[str],
    edges: list[tuple],
    methods: list[str],
) -> go.Figure:
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=methods, horizontal_spacing=0.06, vertical_spacing=0.1
    )

    for method, (row, col) in zip(methods, positions):
        coords, method_name = reduce_dimensions(combined_embeddings, method)
        X, Y = coords.T
        is_first = row == 1 and col == 1

        for label_idx, notice_idx, dist, is_correct in edges:
            label_pt = n_notices + label_idx
            color = (
                "rgba(50, 205, 50, 0.9)"
                if is_correct
                else f"rgba(100, 150, 255, {max(0.2, 1 - dist / 2):.2f})"
            )
            fig.add_trace(
                go.Scatter(
                    x=[X[label_pt], X[notice_idx]],
                    y=[Y[label_pt], Y[notice_idx]],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=3 if is_correct else 1.5,
                        dash="solid" if is_correct else "dot",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=X[:n_notices],
                y=Y[:n_notices],
                mode="markers",
                name="Notices NAF",
                legendgroup="notices",
                showlegend=is_first,
                marker=dict(size=6, color="steelblue", line=dict(width=0.3, color="white")),
                text=notice_hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=X[n_notices:],
                y=Y[n_notices:],
                mode="markers",
                name="Libellés",
                legendgroup="labels",
                showlegend=is_first,
                marker=dict(
                    size=11, color="red", symbol="star", line=dict(width=0.8, color="darkred")
                ),
                text=label_hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(showticklabels=False, title_text=method_name, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(50, 205, 50, 0.9)", width=3),
            name="k-NN correct",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="rgba(100, 150, 255, 0.6)", width=1.5, dash="dot"),
            name="k-NN incorrect",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(height=1000, width=1600, hovermode="closest", plot_bgcolor="white")
    return fig


# %% Per-model evaluation
def evaluate_model(
    model_name: str, eval_df: pd.DataFrame, notices_df: pd.DataFrame, output_dir: str
) -> dict:
    print(f"\n=== {model_name} ===")
    emb_model = build_embedding_model(model_name)

    notice_texts = truncate_texts(notices_df["text_to_embed"].tolist(), MAX_TOKENS)
    notice_codes = notices_df["CODE"].tolist()
    notice_embeddings = np.array(emb_model.embed_documents(notice_texts))
    # Mirrors the "query : {activity}" prefix used at retrieval time in
    # src/neo4j_graph/graph.py:get_closest_codes — required for a fair comparison with
    # instruction-tuned/asymmetric embedding models (query vs. passage formatting).
    query_texts = [f"query : {libelle}" for libelle in eval_df["libelle"]]
    label_embeddings = np.array(emb_model.embed_documents(query_texts))

    indices, distances = compute_knn(
        label_embeddings, notice_embeddings, k=min(K_NN, len(notice_codes))
    )
    metrics, edges = score_retrieval(indices, distances, notice_codes, eval_df["apet2025"].tolist())
    print(json.dumps(metrics, indent=2))

    combined_embeddings = np.vstack([notice_embeddings, label_embeddings])
    notice_hover = [
        f"<b>{code}</b><br>{name}" for code, name in zip(notice_codes, notices_df["NAME"])
    ]
    label_hover = [
        f"<b>{libelle[:80]}</b><br>Cible: {target}"
        for libelle, target in zip(eval_df["libelle"], eval_df["apet2025"])
    ]
    fig = build_comparison_figure(
        combined_embeddings, len(notice_codes), notice_hover, label_hover, edges, REDUCTION_METHODS
    )
    fig.update_layout(
        title_text=f"{model_name} — accuracy@1={metrics['accuracy_at_1']:.1%}, "
        f"recall@{K_NN}={metrics[f'recall_at_{K_NN}']:.1%}"
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    fig.write_html(os.path.join(output_dir, f"{safe_name}_comparison.html"))

    return metrics


# %% Main
def main():
    eval_df = load_eval_sample(EVAL_SAMPLE_PATH)
    notices_df = load_leaf_notices(NAF2025_NOTICES_PATH)
    print(f"Loaded {len(eval_df)} labeled examples and {len(notices_df)} leaf notices")

    summary = {
        model: evaluate_model(model, eval_df, notices_df, OUTPUT_DIR) for model in CANDIDATE_MODELS
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
