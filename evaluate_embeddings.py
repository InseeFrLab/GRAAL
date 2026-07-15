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
import plotly.colors as pcolors
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
from src.utils import storage

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

# NAF2025 taxonomy depth: section (letter) -> division -> group -> class -> sub-class.
LEVEL_NAMES = {1: "SECTION", 2: "DIVISION", 3: "GROUP", 4: "CLASS", 5: "SUBCLASS"}


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


def load_leaf_notices(path: str, include_excludes: bool = False) -> pd.DataFrame:
    """include_excludes=True appends the "Excludes" column to text_to_embed — production's
    build_graph_db.py:37-44 builds the identical NAME+Implementation_rule+Includes+IncludesAlso
    string and never uses Excludes either, even though it's loaded and unused everywhere else.
    Excludes is exactly the NAF-authored "looks similar but isn't" contrastive text, so this
    flag exists to empirically test whether embedding it improves separation between confusable
    categories — it does not change production, only this diagnostic script."""
    full_df = load_notices(path, COLUMNS_TO_KEEP)
    # PARENT_CODE/LEVEL already encode the NAF tree (section=1 ... sub-class=5), so the
    # section a leaf belongs to is found by walking PARENT_CODE up rather than re-deriving
    # it positionally from CODE (the section letter isn't a prefix of the digits).
    ancestors = {row.CODE: row for row in full_df.itertuples()}

    def ancestor_at_level(code: str, target_level: int) -> str | None:
        row = ancestors.get(code)
        while row is not None and row.LEVEL > target_level:
            row = ancestors.get(row.PARENT_CODE)
        return row.CODE if row is not None else None

    df = full_df[full_df["FINAL"] == 1].copy()
    # ancestor_at_level(code, 5) is the identity function for every leaf (a class-level leaf
    # has LEVEL=4<=5, a sub-class leaf has LEVEL=5<=5 — the "row.LEVEL > target_level" guard
    # never fires either way), so SUBCLASS always equals the leaf's own code: it's the one
    # level that can never have >1 member. That's expected, not a bug — the finest meaningful
    # sibling grouping over leaf notices is CLASS (leaves sharing a class parent).
    for level, colname in LEVEL_NAMES.items():
        df[colname] = df["CODE"].map(lambda c, lvl=level: ancestor_at_level(c, lvl))
    df["SECTION_NAME"] = df["SECTION"].map(lambda c: ancestors[c].NAME if c in ancestors else "?")
    df["DIVISION_NAME"] = df["DIVISION"].map(lambda c: ancestors[c].NAME if c in ancestors else "?")
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
    if include_excludes:
        df["text_to_embed"] = df["text_to_embed"] + "\n" + df["Excludes"].fillna("")
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


def compute_hierarchical_accuracy(
    indices: np.ndarray, notice_codes: list[str], target_codes: list[str], notices_df: pd.DataFrame
) -> dict:
    """accuracy@1 only counts an exact leaf-code match — a top-1 that's wrong at the leaf
    but right at section/division/group/class is a much smaller miss than one in a totally
    unrelated branch. This checks the top-1 retrieved notice against the target at every
    taxonomy level, reusing the SECTION/DIVISION/GROUP/CLASS columns already computed by
    load_leaf_notices rather than re-deriving ancestors."""
    code_to_level = {
        colname: dict(zip(notices_df["CODE"], notices_df[colname]))
        for colname in ["SECTION", "DIVISION", "GROUP", "CLASS"]
    }
    n = len(target_codes)
    hits = dict.fromkeys(code_to_level, 0)
    hits["code"] = 0
    for label_idx, target in enumerate(target_codes):
        top1_code = notice_codes[indices[label_idx][0]]
        hits["code"] += int(top1_code == target)
        for colname, mapping in code_to_level.items():
            hits[colname] += int(mapping.get(top1_code) == mapping.get(target))
    return {level.lower(): count / n for level, count in hits.items()}


def compute_worst_offenders(
    eval_df: pd.DataFrame,
    notice_codes: list[str],
    notice_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    indices: np.ndarray,
    top_n: int = 20,
) -> list[dict]:
    """Ranks libellés by similarity to their OWN correct notice (not to whatever was
    actually retrieved) — the lowest-similarity cases are libellés where even the right
    answer doesn't look similar to the query at all, a stronger signal of a genuine
    embedding/labeling problem than "wrong top-1" alone (which could just mean a close
    competitor edged out the correct notice)."""
    code_to_idx = {code: i for i, code in enumerate(notice_codes)}
    norm_labels = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)
    norm_notices = notice_embeddings / np.linalg.norm(notice_embeddings, axis=1, keepdims=True)

    rows = []
    for i, (libelle, target) in enumerate(zip(eval_df["libelle"], eval_df["apet2025"])):
        target_idx = code_to_idx.get(target)
        if target_idx is None:
            continue  # target code isn't among the leaf notices (shouldn't normally happen)
        sim_to_correct = float(norm_labels[i] @ norm_notices[target_idx])
        top1_code = notice_codes[indices[i][0]]
        rows.append(
            {
                "libelle": libelle,
                "cible": target,
                "notice_top1": top1_code,
                "sim_cible": sim_to_correct,
                "correct": top1_code == target,
            }
        )
    rows.sort(key=lambda r: r["sim_cible"])
    return rows[:top_n]


# %% Tree-structure cohesion
def compute_group_cohesion(embeddings: np.ndarray, group_ids: list) -> dict:
    """Do notices sharing a NAF ancestor (e.g. same section/division) actually sit closer
    together in embedding space than random pairs? Compares mean intra-group cosine
    similarity against the overall (any-pair) baseline on the raw embeddings — the 2D
    UMAP/PaCMAP/t-SNE/PCA projections are a lossy visual proxy for this, not the source
    of truth, since a projection can create or hide structure that isn't in the embedding."""
    sims = 1 - cosine_distances(embeddings)
    n = len(embeddings)
    iu = np.triu_indices(n, k=1)
    overall_mean = float(sims[iu].mean())

    groups = pd.Series(group_ids)
    intra_sims = []
    n_groups_with_siblings = 0
    for _, idx in groups.groupby(groups).groups.items():
        members = idx.to_numpy()
        if len(members) < 2:
            continue
        n_groups_with_siblings += 1
        sub = sims[np.ix_(members, members)]
        intra_sims.append(sub[np.triu_indices(len(members), k=1)])

    intra_all = np.concatenate(intra_sims) if intra_sims else np.array([])
    mean_intra = float(intra_all.mean()) if intra_all.size else None
    return {
        "n_groups_with_siblings": n_groups_with_siblings,
        "mean_intra_group_cosine_sim": mean_intra,
        "mean_overall_cosine_sim": overall_mean,
        # >1 means siblings embed closer together than random pairs (tree structure is
        # reflected in the embedding space); ~1 or below means it isn't.
        "cohesion_ratio": (mean_intra / overall_mean) if mean_intra is not None else None,
    }


def compute_group_confusion(embeddings: np.ndarray, group_ids: list) -> tuple[np.ndarray, list]:
    """Group x group mean cosine similarity — the diagonal is each group's own within-group
    cohesion (as in compute_group_cohesion), the off-diagonal is cross-group similarity. A
    cell nearly as bright as its row/column's diagonal is a genuine overlap: two different
    categories the embedding doesn't clearly separate, not just a low overall baseline."""
    sims = 1 - cosine_distances(embeddings)
    groups = pd.Series(group_ids)
    unique_groups = sorted(groups.unique())
    members_by_group = {g: groups[groups == g].index.to_numpy() for g in unique_groups}

    n = len(unique_groups)
    matrix = np.full((n, n), np.nan)
    for i, gi in enumerate(unique_groups):
        idx_i = members_by_group[gi]
        for j, gj in enumerate(unique_groups):
            if j < i:
                matrix[i, j] = matrix[j, i]
                continue
            idx_j = members_by_group[gj]
            if gi == gj:
                if len(idx_i) < 2:
                    continue
                sub = sims[np.ix_(idx_i, idx_i)]
                vals = sub[np.triu_indices(len(idx_i), k=1)]
            else:
                vals = sims[np.ix_(idx_i, idx_j)].ravel()
            matrix[i, j] = vals.mean() if vals.size else np.nan
    return matrix, unique_groups


def top_confusable_pairs(matrix: np.ndarray, group_labels: list, top_n: int = 15) -> list[dict]:
    """Rank off-diagonal group pairs by cross-similarity relative to their own within-group
    cohesion — a pair whose cross-similarity approaches (or exceeds) either group's own
    cohesion is a stronger overlap signal than raw cross-similarity alone, since some groups
    are just diffuse overall and would otherwise dominate a raw-similarity ranking."""
    n = len(group_labels)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            cross = matrix[i, j]
            within_i, within_j = matrix[i, i], matrix[j, j]
            if np.isnan(cross) or np.isnan(within_i) or np.isnan(within_j):
                continue
            rows.append(
                {
                    "group_a": group_labels[i],
                    "group_b": group_labels[j],
                    "cross_sim": float(cross),
                    "within_a": float(within_i),
                    "within_b": float(within_j),
                    "confusability": float(cross / min(within_i, within_j)),
                }
            )
    rows.sort(key=lambda r: r["confusability"], reverse=True)
    return rows[:top_n]


def compute_query_passage_gap(notice_embeddings: np.ndarray, label_embeddings: np.ndarray) -> dict:
    """Quantifies whether libellés (queries) and notices (passages) occupy visibly different
    regions of the embedding space — a known property of bi-encoder retrieval models, which
    are trained to rank the correct passage above others (relative), not to co-locate queries
    and passages in absolute coordinates. within_label_sim >> label_to_notice_sim indicates
    labels cluster with each other more than with their matching notices — i.e. a real gap,
    not just a projection artifact of the 2D scatter plots."""
    within_notice = 1 - cosine_distances(notice_embeddings)
    within_label = 1 - cosine_distances(label_embeddings)
    cross = 1 - cosine_distances(label_embeddings, notice_embeddings)

    notice_centroid = notice_embeddings.mean(axis=0, keepdims=True)
    label_centroid = label_embeddings.mean(axis=0, keepdims=True)
    centroid_sim = float(1 - cosine_distances(label_centroid, notice_centroid)[0, 0])

    n_notices, n_labels = len(notice_embeddings), len(label_embeddings)
    return {
        "within_notice_sim": float(within_notice[np.triu_indices(n_notices, k=1)].mean()),
        "within_label_sim": float(within_label[np.triu_indices(n_labels, k=1)].mean()),
        "label_to_notice_sim": float(cross.mean()),
        "centroid_similarity": centroid_sim,
    }


LEVEL_LABELS = {
    "SECTION": "Section",
    "DIVISION": "Division",
    "GROUP": "Groupe",
    "CLASS": "Classe",
    "SUBCLASS": "Sous-classe",
}


def build_similarity_heatmap(notice_embeddings: np.ndarray, notices_df: pd.DataFrame) -> go.Figure:
    """Pairwise cosine similarity between notices, at full leaf resolution and sorted by
    taxonomy path (section->division->group->class->code) — nested block sizes then
    visually correspond to nested tree levels. Computed on the raw embeddings, so unlike
    the 2D scatter projections it can't create or hide structure that isn't really there."""
    order = notices_df.sort_values(
        ["SECTION", "DIVISION", "GROUP", "CLASS", "CODE"]
    ).index.to_numpy()
    sorted_codes = notices_df["CODE"].to_numpy()[order]
    sorted_sections = notices_df["SECTION"].to_numpy()[order]

    sims = 1 - cosine_distances(notice_embeddings[order])
    np.fill_diagonal(sims, np.nan)  # self-similarity is always 1.0 and would wash out the scale

    off_diag = sims[~np.isnan(sims)]
    zmin, zmax = np.quantile(off_diag, [0.02, 0.98])

    # Plotly serializes numeric arrays as base64-encoded typed-array binary ("bdata"), which
    # ignores decimal rounding (it always stores the full width of whatever dtype it's given)
    # — float64 here would be ~7.5MB for 747x747; float32 halves it with no visible precision
    # loss (3-4 significant digits is already far finer than perceptible color differences).
    fig = go.Figure(
        go.Heatmap(
            z=sims.astype(np.float32),
            x=sorted_codes,
            y=sorted_codes,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="cosinus"),
            hoverongaps=False,
        )
    )
    fig.update_xaxes(
        showticklabels=False, title_text="Notices (triées section→division→groupe→classe)"
    )
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    n = len(sorted_codes)
    for b in np.where(sorted_sections[1:] != sorted_sections[:-1])[0] + 0.5:
        fig.add_shape(
            type="line",
            x0=b,
            x1=b,
            y0=-0.5,
            y1=n - 0.5,
            line=dict(color="black", width=0.5),
            opacity=0.35,
            layer="above",
        )
        fig.add_shape(
            type="line",
            y0=b,
            y1=b,
            x0=-0.5,
            x1=n - 0.5,
            line=dict(color="black", width=0.5),
            opacity=0.35,
            layer="above",
        )
    fig.update_layout(
        title="Similarité cosinus entre notices (espace haute dimension, non projeté) "
        "— lignes noires = frontières de section"
    )
    return fig


def build_level_cohesion_figure(cohesion_by_level: dict) -> go.Figure:
    """One bar per taxonomy level (not a line — the levels are discrete/qualitative, a line
    would wrongly imply interpolation between them), labeled with its group count so a bar
    backed by a handful of groups isn't visually mistaken for one backed by hundreds."""
    labels = [LEVEL_LABELS[colname.upper()] for colname in cohesion_by_level]
    ratios = [v["cohesion_ratio"] for v in cohesion_by_level.values()]
    counts = [v["n_groups_with_siblings"] for v in cohesion_by_level.values()]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=ratios,
            text=[f"n={c} groupes" if c else "aucun groupe ≥2" for c in counts],
            textposition="outside",
            marker_color="steelblue",
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="baseline (aucune structure)",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Cohésion intra-groupe par niveau de granularité (siblings vs. paires aléatoires)",
        yaxis_title="ratio de cohésion (>1 = siblings plus proches que des paires aléatoires)",
    )
    return fig


def build_hierarchical_accuracy_figure(hierarchical_accuracy: dict) -> go.Figure:
    """One bar per taxonomy level (code = exact leaf) showing how often the top-1 retrieved
    notice matches the target at that level — since a coarser level is always at least as
    easy to get right as a finer one, bars should increase monotonically toward Section;
    a dip would indicate something structurally off rather than just gradual difficulty."""
    order = ["code", "class", "group", "division", "section"]
    labels = [LEVEL_LABELS[level.upper()] if level != "code" else "Code exact" for level in order]
    values = [hierarchical_accuracy[level] for level in order]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title="Précision du top-1 par niveau de granularité",
        yaxis_title="précision (le retrieved top-1 correspond-il à la cible à ce niveau ?)",
        yaxis_tickformat=".0%",
    )
    return fig


def build_similarity_distribution_figure(edges: list[tuple]) -> go.Figure:
    """Distribution (not just the two means already in mean_cosine_sim_correct/incorrect) of
    k-NN cosine similarity, split by correct vs. incorrect — reveals whether there's a usable
    confidence threshold (little overlap) or the two distributions are hard to tell apart
    (heavy overlap), which is what actually determines if a similarity cutoff could flag
    low-confidence retrievals for the downstream LLM navigator rather than trusting top-1."""
    correct_sims = [1 - dist for _, _, dist, is_correct in edges if is_correct]
    incorrect_sims = [1 - dist for _, _, dist, is_correct in edges if not is_correct]

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=correct_sims,
            name="Correct",
            box_visible=True,
            meanline_visible=True,
            line_color="rgba(50, 130, 50, 0.9)",
            fillcolor="rgba(50, 205, 50, 0.3)",
        )
    )
    fig.add_trace(
        go.Violin(
            y=incorrect_sims,
            name="Incorrect",
            box_visible=True,
            meanline_visible=True,
            line_color="rgba(70, 110, 180, 0.9)",
            fillcolor="rgba(100, 150, 255, 0.3)",
        )
    )
    fig.update_layout(
        title="Distribution des similarités cosinus : k-NN corrects vs. incorrects",
        yaxis_title="similarité cosinus",
    )
    return fig


def compute_retrieval_confusion(
    edges: list[tuple], target_codes: list[str], notice_codes: list[str], code_to_group: dict
) -> list[dict]:
    """Tally (true group -> wrongly-retrieved group) over the eval sample's incorrect k-NN
    edges — grounded in actual retrieval failures rather than embedding proximity alone, but
    noisier than compute_group_confusion since it's limited to whatever's in the eval sample.
    Same-group misses (wrong leaf, right section/division) are excluded — that's a fine-
    grained miss, not the cross-category confusion this is meant to surface."""
    counts: dict[tuple, int] = {}
    for label_idx, notice_idx, _dist, is_correct in edges:
        if is_correct:
            continue
        true_group = code_to_group.get(target_codes[label_idx])
        retrieved_group = code_to_group.get(notice_codes[notice_idx])
        if true_group is None or retrieved_group is None or true_group == retrieved_group:
            continue
        key = (true_group, retrieved_group)
        counts[key] = counts.get(key, 0) + 1
    rows = [
        {"true_group": tg, "retrieved_group": rg, "count": c}
        for (tg, rg), c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return rows


def build_confusion_heatmap(matrix: np.ndarray, group_labels: list, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=group_labels,
            y=group_labels,
            colorscale="Viridis",
            colorbar=dict(title="cosinus"),
            zmin=float(np.nanmin(matrix)),
            zmax=float(np.nanmax(matrix)),
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return fig


def render_table_html(rows: list[dict], columns: list[tuple]) -> str:
    """columns: list of (key, header_label) — used for both confusable-pairs and retrieval-
    confusion tables. Floats are rounded for readability, other values printed as-is."""
    if not rows:
        return "<p><em>Aucune donnée.</em></p>"
    header = "".join(f"<th>{label}</th>" for _, label in columns)
    body_rows = []
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r[key]
            cells.append(f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{v}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="confusion-table"><thead><tr>'
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


PROJECTION_CAPTION = (
    "UMAP et PaCMAP préservent mieux la structure locale (cohésion de voisinage) que t-SNE, "
    "qui déforme les distances relatives et la taille apparente des clusters, ou que PCA, une "
    "projection linéaire peu apte à révéler une séparation non-linéaire des groupes. Pour juger "
    "de la cohésion réelle des clusters, privilégier les panneaux UMAP/PaCMAP plutôt que t-SNE/PCA. "
    "Note : les libellés (étoiles) peuvent visuellement sembler éloignés de leur notice correcte — "
    'voir la section "Écart requête / passage" plus bas avant d\'interpréter cela comme un défaut.'
)

QUERY_PASSAGE_CAPTION = (
    "De nombreux modèles d'embedding de recherche (bi-encodeurs) sont entraînés pour classer la "
    "bonne passage devant les autres (ranking relatif), pas pour placer requêtes et passages au "
    "même endroit dans l'espace vectoriel (position absolue) — un écart entre les deux nuages de "
    "points est donc normal et ne signifie pas que la recherche fonctionne mal. Le signal fiable "
    "est ici within_label_sim vs. label_to_notice_sim : si les libellés sont bien plus proches "
    "entre eux qu'ils ne le sont de leur notice correcte, c'est la preuve chiffrée de cet écart — "
    "la bonne façon de juger la recherche reste alors les arêtes k-NN (relatif), pas la position "
    "visuelle d'une étoile par rapport aux points de sa couleur (absolu)."
)

CONFUSION_CAPTION = (
    "Contrairement à la cohésion (qui ne regarde que l'intérieur de chaque groupe), cette carte "
    "montre la similarité moyenne ENTRE groupes différents (hors diagonale). Une cellule presque "
    "aussi claire que la diagonale de sa ligne/colonne indique un chevauchement réel : deux "
    "catégories que l'embedding ne sépare pas nettement — à vérifier si c'est un défaut de "
    "l'embedding ou une ambiguïté propre à la nomenclature elle-même."
)


def render_report_html(model_name: str, metrics: dict, sections: list[tuple]) -> str:
    """Combine several Plotly figures and/or raw HTML blocks (e.g. tables) + captions into
    one standalone HTML file. sections: (heading, caption, content) where content is either
    a go.Figure or an HTML string. Only the first figure loads plotly.js (via CDN); later
    figures are rendered without it and reuse the same page-level include — the CDN
    <script> tag has no async/defer so it blocks before any later inline Plotly.newPlot()
    call runs, and each to_html() call emits a fresh random div id, so concatenation is safe."""
    parts = [
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{model_name} — embedding diagnostics</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1400px; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; border-top: 1px solid #ddd; padding-top: 1rem; }}
  .caption {{ color: #555; font-size: 0.9rem; max-width: 900px; }}
  pre {{ background: #f6f6f6; padding: 0.75rem; overflow-x: auto; }}
  table.confusion-table {{ border-collapse: collapse; margin-top: 0.5rem; }}
  table.confusion-table th, table.confusion-table td {{
    border-bottom: 1px solid #ddd; padding: 0.3rem 0.75rem; text-align: right; font-size: 0.9rem;
  }}
  table.confusion-table th:first-child, table.confusion-table td:first-child {{ text-align: left; }}
</style></head><body>
<h1>{model_name}</h1>
<p>accuracy@1={metrics["accuracy_at_1"]:.1%}, recall@{K_NN}={metrics[f"recall_at_{K_NN}"]:.1%}</p>
"""
    ]
    plotlyjs_loaded = False
    for heading, caption, content in sections:
        parts.append(f"<h2>{heading}</h2>")
        if caption:
            parts.append(f'<p class="caption">{caption}</p>')
        if isinstance(content, go.Figure):
            parts.append(
                content.to_html(
                    full_html=False,
                    include_plotlyjs="cdn" if not plotlyjs_loaded else False,
                    config={"responsive": True},
                )
            )
            plotlyjs_loaded = True
        else:
            parts.append(content)  # raw HTML (e.g. a table) — not every section is a figure
    parts.append(
        f"<h2>Métriques complètes</h2><pre>{json.dumps(metrics, indent=2, ensure_ascii=False)}</pre>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


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
def _edge_segments(
    edges: list[tuple], X: np.ndarray, Y: np.ndarray, n_notices: int, want_correct: bool
) -> tuple[list, list]:
    """Flatten the wanted edges into a single None-separated x/y run, so they can be
    drawn as one Scatter trace instead of one trace per edge (which balloons the
    written HTML to hundreds of traces and makes it slow to load/read)."""
    xs, ys = [], []
    for label_idx, notice_idx, dist, is_correct in edges:
        if is_correct != want_correct:
            continue
        label_pt = n_notices + label_idx
        xs.extend([X[label_pt], X[notice_idx], None])
        ys.extend([Y[label_pt], Y[notice_idx], None])
    return xs, ys


def build_comparison_figure(
    combined_embeddings: np.ndarray,
    n_notices: int,
    notice_hover: list[str],
    label_hover: list[str],
    edges: list[tuple],
    methods: list[str],
    notice_section: list[str],
    section_color: dict[str, str],
    section_legend: dict[str, str],
) -> go.Figure:
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=methods, horizontal_spacing=0.06, vertical_spacing=0.1
    )

    section_indices: dict[str, list[int]] = {}
    for i, sec in enumerate(notice_section):
        section_indices.setdefault(sec, []).append(i)

    for method, (row, col) in zip(methods, positions):
        coords, method_name = reduce_dimensions(combined_embeddings, method)
        X, Y = coords.T
        is_first = row == 1 and col == 1

        incorrect_x, incorrect_y = _edge_segments(edges, X, Y, n_notices, want_correct=False)
        fig.add_trace(
            go.Scatter(
                x=incorrect_x,
                y=incorrect_y,
                mode="lines",
                line=dict(color="rgba(100, 150, 255, 0.5)", width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        correct_x, correct_y = _edge_segments(edges, X, Y, n_notices, want_correct=True)
        fig.add_trace(
            go.Scatter(
                x=correct_x,
                y=correct_y,
                mode="lines",
                line=dict(color="rgba(50, 205, 50, 0.9)", width=3, dash="solid"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        for sec, idxs in section_indices.items():
            idx_arr = np.array(idxs)
            fig.add_trace(
                go.Scatter(
                    x=X[idx_arr],
                    y=Y[idx_arr],
                    mode="markers",
                    name=section_legend.get(sec, str(sec)),
                    legendgroup=f"section-{sec}",
                    showlegend=is_first,
                    marker=dict(
                        size=6,
                        color=section_color.get(sec, "steelblue"),
                        line=dict(width=0.3, color="white"),
                    ),
                    text=[notice_hover[i] for i in idxs],
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
    fig.update_layout(autosize=True, height=1000, hovermode="closest", plot_bgcolor="white")
    return fig


# %% Per-model evaluation
def compute_model_diagnostics(
    model_name: str,
    eval_df: pd.DataFrame,
    notices_df: pd.DataFrame,
    run_label: str | None = None,
    notice_embeddings: np.ndarray | None = None,
) -> dict:
    """The compute phase only: embeddings, retrieval scoring, cohesion/confusion/gap
    metrics, and the raw data figure-builders need — no HTML/figure construction. Split
    out from evaluate_model so this can be reused from contexts other than the CLI script
    (e.g. a Quarto report) without re-running the (slow, API-calling) computation twice or
    duplicating this logic. run_label distinguishes console/report naming from the actual
    model_name passed to the embedding API — used to run the same model against multiple
    notices_df variants (e.g. baseline vs. with Excludes appended) without re-tagging the
    underlying embedding model itself. notice_embeddings can be passed in to skip re-
    embedding notices_df — e.g. reuse the embeddings from a small eval_df run (used for a
    readable plot) when computing metrics again against a much larger eval_df (used for
    statistically sturdier accuracy/recall/confusion numbers): notices_df doesn't change
    between those two calls, only eval_df does, so there's no need to pay for the (far more
    expensive, since there are many more notices than labels) notice embedding call twice."""
    run_label = run_label or model_name
    print(f"\n=== {run_label} ===")
    emb_model = build_embedding_model(model_name)

    notice_codes = notices_df["CODE"].tolist()
    if notice_embeddings is None:
        notice_texts = truncate_texts(notices_df["text_to_embed"].tolist(), MAX_TOKENS)
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

    hierarchical_accuracy = compute_hierarchical_accuracy(
        indices, notice_codes, eval_df["apet2025"].tolist(), notices_df
    )
    print("Hierarchical accuracy:", json.dumps(hierarchical_accuracy, indent=2))
    metrics["hierarchical_accuracy"] = hierarchical_accuracy

    worst_offenders = compute_worst_offenders(
        eval_df, notice_codes, notice_embeddings, label_embeddings, indices
    )
    metrics["worst_offenders"] = worst_offenders

    cohesion_by_level = {}
    for colname in LEVEL_NAMES.values():
        cohesion = compute_group_cohesion(notice_embeddings, notices_df[colname].tolist())
        print(f"Sibling cohesion ({colname}):", json.dumps(cohesion, indent=2))
        cohesion_by_level[colname.lower()] = cohesion
    metrics["sibling_cohesion"] = cohesion_by_level

    query_passage_gap = compute_query_passage_gap(notice_embeddings, label_embeddings)
    print("Query/passage gap:", json.dumps(query_passage_gap, indent=2))
    metrics["query_passage_gap"] = query_passage_gap

    section_matrix, section_labels = compute_group_confusion(
        notice_embeddings, notices_df["SECTION"].tolist()
    )
    division_matrix, division_labels = compute_group_confusion(
        notice_embeddings, notices_df["DIVISION"].tolist()
    )
    top_confusable_sections = top_confusable_pairs(section_matrix, section_labels)
    top_confusable_divisions = top_confusable_pairs(division_matrix, division_labels)
    metrics["top_confusable_sections"] = top_confusable_sections
    metrics["top_confusable_divisions"] = top_confusable_divisions

    code_to_section = dict(zip(notices_df["CODE"], notices_df["SECTION"]))
    retrieval_confusion = compute_retrieval_confusion(
        edges, eval_df["apet2025"].tolist(), notice_codes, code_to_section
    )
    metrics["retrieval_confusion_sections"] = retrieval_confusion

    notice_hover = [
        f"<b>{code}</b><br>{name}<br><i>{section}</i>"
        for code, name, section in zip(notice_codes, notices_df["NAME"], notices_df["SECTION_NAME"])
    ]
    label_hover = [
        f"<b>{libelle[:80]}</b><br>Cible: {target}"
        for libelle, target in zip(eval_df["libelle"], eval_df["apet2025"])
    ]

    notice_section = notices_df["SECTION"].tolist()
    unique_sections = sorted(set(notice_section))
    palette = pcolors.qualitative.Alphabet
    section_color = {sec: palette[i % len(palette)] for i, sec in enumerate(unique_sections)}
    section_legend = dict(zip(notices_df["SECTION"], notices_df["SECTION_NAME"].str[:28]))

    return {
        "run_label": run_label,
        "metrics": metrics,
        "notice_embeddings": notice_embeddings,
        "label_embeddings": label_embeddings,
        "combined_embeddings": np.vstack([notice_embeddings, label_embeddings]),
        "notice_codes": notice_codes,
        "edges": edges,
        "hierarchical_accuracy": hierarchical_accuracy,
        "worst_offenders": worst_offenders,
        "cohesion_by_level": cohesion_by_level,
        "query_passage_gap": query_passage_gap,
        "section_matrix": section_matrix,
        "section_labels": section_labels,
        "division_matrix": division_matrix,
        "division_labels": division_labels,
        "top_confusable_sections": top_confusable_sections,
        "top_confusable_divisions": top_confusable_divisions,
        "retrieval_confusion": retrieval_confusion,
        "notice_hover": notice_hover,
        "label_hover": label_hover,
        "notice_section": notice_section,
        "section_color": section_color,
        "section_legend": section_legend,
    }


def evaluate_model(
    model_name: str,
    eval_df: pd.DataFrame,
    notices_df: pd.DataFrame,
    output_dir: str,
    run_label: str | None = None,
) -> dict:
    diag = compute_model_diagnostics(model_name, eval_df, notices_df, run_label)
    run_label = diag["run_label"]
    metrics = diag["metrics"]

    comparison_fig = build_comparison_figure(
        diag["combined_embeddings"],
        len(diag["notice_codes"]),
        diag["notice_hover"],
        diag["label_hover"],
        diag["edges"],
        REDUCTION_METHODS,
        diag["notice_section"],
        diag["section_color"],
        diag["section_legend"],
    )
    comparison_fig.update_layout(
        title_text=f"{run_label} — accuracy@1={metrics['accuracy_at_1']:.1%}, "
        f"recall@{K_NN}={metrics[f'recall_at_{K_NN}']:.1%}"
    )
    hierarchical_accuracy_fig = build_hierarchical_accuracy_figure(diag["hierarchical_accuracy"])
    distribution_fig = build_similarity_distribution_figure(diag["edges"])
    worst_offenders_html = render_table_html(
        diag["worst_offenders"],
        [
            ("libelle", "Libellé"),
            ("cible", "Cible"),
            ("notice_top1", "Notice récupérée (top-1)"),
            ("sim_cible", "Similarité à la cible"),
            ("correct", "Top-1 correct ?"),
        ],
    )
    heatmap_fig = build_similarity_heatmap(diag["notice_embeddings"], notices_df)
    level_fig = build_level_cohesion_figure(diag["cohesion_by_level"])
    confusion_fig = build_confusion_heatmap(
        diag["section_matrix"],
        diag["section_labels"],
        "Confusion inter-sections (similarité moyenne)",
    )
    section_pairs_html = render_table_html(
        diag["top_confusable_sections"],
        [
            ("group_a", "Section A"),
            ("group_b", "Section B"),
            ("cross_sim", "Similarité croisée"),
            ("within_a", "Cohésion A"),
            ("within_b", "Cohésion B"),
            ("confusability", "Indice de confusion"),
        ],
    )
    division_pairs_html = render_table_html(
        diag["top_confusable_divisions"],
        [
            ("group_a", "Division A"),
            ("group_b", "Division B"),
            ("cross_sim", "Similarité croisée"),
            ("within_a", "Cohésion A"),
            ("within_b", "Cohésion B"),
            ("confusability", "Indice de confusion"),
        ],
    )
    retrieval_confusion_html = render_table_html(
        diag["retrieval_confusion"],
        [
            ("true_group", "Section réelle"),
            ("retrieved_group", "Section récupérée (erreur)"),
            ("count", "Occurrences"),
        ],
    )
    query_passage_gap = diag["query_passage_gap"]
    query_passage_html = f"""<ul>
<li>Similarité intra-notices : {query_passage_gap["within_notice_sim"]:.3f}</li>
<li>Similarité intra-libellés : {query_passage_gap["within_label_sim"]:.3f}</li>
<li>Similarité libellé → notice (toutes paires) : {query_passage_gap["label_to_notice_sim"]:.3f}</li>
<li>Similarité libellé → notice correcte : {metrics["mean_cosine_sim_correct"]:.3f}</li>
<li>Similarité des centroïdes (direction moyenne libellés vs. notices) : {query_passage_gap["centroid_similarity"]:.3f}</li>
</ul>"""

    sections = [
        ("Projections 2D (UMAP / PaCMAP / t-SNE / PCA)", PROJECTION_CAPTION, comparison_fig),
        ("Précision par niveau hiérarchique", None, hierarchical_accuracy_fig),
        ("Distribution des similarités (correct vs. incorrect)", None, distribution_fig),
        ("Cas les plus problématiques", None, worst_offenders_html),
        ("Similarité cosinus entre notices (triée par arbre taxonomique)", None, heatmap_fig),
        ("Cohésion intra-groupe par niveau de granularité", None, level_fig),
        ("Confusion entre sections (chevauchement)", CONFUSION_CAPTION, confusion_fig),
        ("Paires de sections les plus confondues", None, section_pairs_html),
        ("Paires de divisions les plus confondues", None, division_pairs_html),
        (
            "Confusions observées en récupération (échantillon éval, n=30 — indicatif seulement)",
            None,
            retrieval_confusion_html,
        ),
        (
            "Écart requête / passage (libellés vs. notices)",
            QUERY_PASSAGE_CAPTION,
            query_passage_html,
        ),
    ]
    html = render_report_html(run_label, metrics, sections)

    storage.makedirs(output_dir)
    safe_name = run_label.replace("/", "_")
    comparison_path = os.path.join(output_dir, f"{safe_name}_comparison.html")
    with storage.open_path(comparison_path, "w", encoding="utf-8") as f:
        f.write(html)

    return metrics


def print_excludes_comparison(baseline: dict, with_excludes: dict) -> None:
    print("\n--- Excludes-field experiment: baseline vs. with Excludes ---")
    for key in ["accuracy_at_1", f"recall_at_{K_NN}"]:
        print(f"{key}: {baseline[key]:.1%} -> {with_excludes[key]:.1%}")
    for level in LEVEL_NAMES.values():
        level = level.lower()
        b = baseline["sibling_cohesion"][level]["cohesion_ratio"]
        w = with_excludes["sibling_cohesion"][level]["cohesion_ratio"]
        b_str = f"{b:.3f}" if b is not None else "n/a"
        w_str = f"{w:.3f}" if w is not None else "n/a"
        print(f"cohesion_ratio ({level}): {b_str} -> {w_str}")


# %% Main
def main():
    eval_df = load_eval_sample(EVAL_SAMPLE_PATH)
    notices_df = load_leaf_notices(NAF2025_NOTICES_PATH)
    notices_df_excludes = load_leaf_notices(NAF2025_NOTICES_PATH, include_excludes=True)
    print(f"Loaded {len(eval_df)} labeled examples and {len(notices_df)} leaf notices")

    summary = {}
    for model in CANDIDATE_MODELS:
        baseline = evaluate_model(
            model, eval_df, notices_df, OUTPUT_DIR, run_label=f"{model}_baseline"
        )
        with_excludes = evaluate_model(
            model, eval_df, notices_df_excludes, OUTPUT_DIR, run_label=f"{model}_with_excludes"
        )
        print_excludes_comparison(baseline, with_excludes)
        summary[model] = {"baseline": baseline, "with_excludes": with_excludes}

    storage.makedirs(OUTPUT_DIR)
    with storage.open_path(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
