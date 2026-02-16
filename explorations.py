
# %% Reload les codes
%load_ext autoreload
%autoreload 2


# %% Imports

import os
import numpy as np
import plotly.graph_objects as go
from langchain_openai import OpenAIEmbeddings
import polars as pl
import umap
import pacmap
from plotly.subplots import make_subplots

import s3fs
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import neo4j_config
from src.neo4j_graph.graph import Graph
from src.main import classify_navigator



# %% Config
N_CODES = 20
K_NN = 1

PATH = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
COLUMNS = ["libelle", "nace2025"]

REDUCTION_METHOD = "umap"  # Options: "umap", "pacmap", "tsne", "pca"

emb_model = OpenAIEmbeddings(
        model=os.environ['EMBEDDING_MODEL'],
        openai_api_base=os.environ['URL_EMBEDDING_API'],
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.environ["AWS_ACCESS_KEY_ID"], 
    secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
    token=os.environ["AWS_SESSION_TOKEN"])

graph = Graph(neo4j_config)


# %% Récupération des données
query = """
MATCH path = (root)-[*]->(n)
WHERE n.FINAL = 1
  AND n.embedding IS NOT NULL
  AND root.LEVEL = 0
RETURN n.embedding as embedding,
       n.NAME as name,
       n.CODE as code, 
       [node IN nodes(path) | node.CODE] as path_codes,
       [node IN nodes(path) | node.LEVEL] as path_levels
"""

results = graph.graph.query(query)

embeddings = []
names = []
paths = []
codes_dict = {}

for idx, record in enumerate(results):
    embeddings.append(record["embedding"])
    names.append(record["name"])
    code = record["code"]
    code_clean = code.replace(".", "").replace(" ", "")
    codes_dict[code_clean] = idx
    
    path_str = " → ".join([
        name for lvl, name in zip(record["path_levels"], record["path_codes"])
    ])
    paths.append(path_str)

n_nace_nodes = len(embeddings)



# %%
import os
import s3fs

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])



def sample_codes(fs: s3fs.S3FileSystem, population_path: str, code_column: str, n_codes: int):
    """
    Sample codes using Polars from S3.
    
    Args:
        fs: S3FileSystem configuré
        population_path: chemin S3 (avec ou sans s3://)
        code_column: nom de la colonne
        n_codes: nombre de codes à échantillonner
    """
    with fs.open(population_path, 'rb') as f:
        df = pl.read_parquet(f)

    sampled = df.select(code_column).sample(n=n_codes, with_replacement=True)
    
    return sampled[code_column].to_numpy()

    label_idx = n_nace_nodes + i
    if target_code in codes_dict: 
        label_to_code_idx[label_idx] = codes_dict[target_code]

codes = sample_codes(
    fs=fs,
    population_path=PATH,
    code_column=COLUMNS, 
    n_codes=N_CODES)

labels, target_codes = zip(*codes)
labels_embeddings = emb_model.embed_documents(list(labels))
label_to_code_idx = {}

for i, (label, label_emb, target_code) in enumerate(zip(labels, labels_embeddings, target_codes)):
    embeddings.append(label_emb)
    names.append(label[:50])
    paths.append(f"Libellé -> Code cible: {target_code}")

    label_idx = n_nace_nodes + i
    if target_code in codes_dict: 
        label_to_code_idx[label_idx] = codes_dict[target_code]

embeddings = np.array(embeddings)


# %% Classification by the navigator
from src.config import neo4j_config
from src.neo4j_graph.graph import Graph
from src.main import classify_navigator

graph = Graph(neo4j_config)

results = await classify_navigator(labels)
codes = [result.code.replace(".", "").replace(" ", "") for result in results]
print(codes)



# %% Calcul des k plus proches voisins parmi les codes NACE
def compute_knn_to_nace_codes(label_embeddings, nace_embeddings, label_start_idx, k=5):
    """
    Pour chaque libellé, trouve les k codes NACE les plus proches dans l'espace des embeddings.
    
    Args:
        label_embeddings: Embeddings des libellés (n_labels, dim)
        nace_embeddings: Embeddings des codes NACE (n_nace, dim)
        coords: Coordonnées UMAP de tous les points (n_total, 2)
        label_start_idx: Index de début des libellés (= n_nace_nodes)
        k: Nombre de voisins NACE à trouver
    
    Returns:
        Liste de tuples (label_idx, nace_idx, distance_cosine)
    """
    # Calculer k-NN uniquement parmi les codes NACE
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(nace_embeddings)
    
    edges = []
    for i, label_emb in enumerate(label_embeddings):
        distances, indices = nbrs.kneighbors([label_emb])
        
        label_idx = label_start_idx + i
        for nace_idx, dist in zip(indices[0], distances[0]):
            edges.append((label_idx, nace_idx, dist))
    
    return edges


# Séparer les embeddings NACE et libellés
nace_embeddings = embeddings[:n_nace_nodes]
label_embeddings = embeddings[n_nace_nodes:]

# Calculer les k plus proches codes NACE pour chaque libellé
knn_edges = compute_knn_to_nace_codes(
    label_embeddings=label_embeddings,
    nace_embeddings=nace_embeddings,
    label_start_idx=n_nace_nodes,
    k=K_NN
)

print(f"Edges k-NN calculés: {len(knn_edges)}")
print(f"Chaque libellé est relié à {K_NN} codes NACE")



# %% Réduction dimensionnelle - CHOISIR LA MÉTHODE
def reduce_dimensions(embeddings, method="umap", random_state=42):
    """
    Réduit les dimensions des embeddings avec différentes méthodes.
    
    Args:
        embeddings: Array (n_samples, n_features)
        method: "umap", "pacmap", "tsne", ou "pca"
        random_state: Seed aléatoire
    
    Returns:
        coords: Array (n_samples, 2)
        method_name: Nom de la méthode utilisée
    """
    print(f"🔄 Réduction dimensionnelle avec {method.upper()}...")
    
    if method == "umap":
        reducer = umap.UMAP(
            random_state=random_state,
            n_neighbors=10,
            min_dist=0.1,
            metric='cosine',
            spread=1.0
        )
        coords = reducer.fit_transform(embeddings)
        return coords, "UMAP (cosine)"
    
    elif method == "pacmap":
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            distance='angular',
            random_state=random_state
        )
        coords = reducer.fit_transform(embeddings)
        return coords, "PaCMAP (cosine)"

    elif method == "tsne":
        # t-SNE avec distance cosine via precomputed distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
        
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,
            metric='cosine',
            max_iter=1000
        )
        coords = reducer.fit_transform(distances)
        return coords, "t-SNE (cosine)"
    
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        variance_explained = reducer.explained_variance_ratio_.sum()
        return coords, f"PCA (variance: {variance_explained:.1%})"
    
    else:
        raise ValueError(f"Méthode inconnue: {method}. Utilisez 'umap', 'pacmap', 'tsne', ou 'pca'")


# Réduire les dimensions
coords, method_name = reduce_dimensions(embeddings, method=REDUCTION_METHOD)
X, Y = coords.T

print(f"✓ Réduction terminée avec {method_name}")
# %% Annoter visuellement les prédictions correctes

fig = go.Figure()

# Identifier les prédictions correctes
correct_predictions = []
for label_idx, nace_idx, dist in knn_edges:
    true_code = label_to_code_idx.get(label_idx)
    if true_code == nace_idx:
        correct_predictions.append((label_idx, nace_idx, dist))

# 1. Lignes k-NN INCORRECTES (bleu standard)
for label_idx, nace_idx, dist in knn_edges:
    if (label_idx, nace_idx, dist) not in correct_predictions:
        alpha = max(0.2, 1 - dist/2)
        fig.add_trace(go.Scatter(
            x=[X[label_idx], X[nace_idx]],
            y=[Y[label_idx], Y[nace_idx]],
            mode='lines',
            line=dict(color=f'rgba(100, 150, 255, {alpha})', width=1.5, dash='dot'),
            showlegend=False,
            hovertemplate=f'k-NN (incorrect)<br>Sim: {1-dist:.3f}<extra></extra>'
        ))

# 2. Lignes k-NN CORRECTES (vert épais)
for label_idx, nace_idx, dist in correct_predictions:
    fig.add_trace(go.Scatter(
        x=[X[label_idx], X[nace_idx]],
        y=[Y[label_idx], Y[nace_idx]],
        mode='lines',
        line=dict(color='rgba(50, 205, 50, 0.9)', width=4),  # Vert épais
        name='k-NN correct' if label_idx == correct_predictions[0][0] else '',
        legendgroup='knn_correct',
        showlegend=(label_idx == correct_predictions[0][0]),
        hovertemplate=f'k-NN ✓ CORRECT<br>Sim: {1-dist:.3f}<extra></extra>'
    ))

# 3. Ground truth (rouge)
for label_idx, code_idx in label_to_code_idx.items():
    fig.add_trace(go.Scatter(
        x=[X[label_idx], X[code_idx]],
        y=[Y[label_idx], Y[code_idx]],
        mode='lines',
        line=dict(color='rgba(255, 100, 100, 0.7)', width=2, dash='solid'),
        showlegend=False,
        hovertemplate='Ground Truth<extra></extra>'
    ))

# 3. Ajouter les nœuds NACE (cercles)
fig.add_trace(go.Scatter(
    x=X[:n_nace_nodes], 
    y=Y[:n_nace_nodes],
    mode='markers',
    name='Codes NACE',
    marker=dict(
        size=10,
        color=np.arange(n_nace_nodes),
        colorscale='Viridis',
        showscale=True,
        line=dict(width=0.5, color='white'),
        symbol='circle'
    ),
    text=[f"<b>{name}</b><br><br>{path}" for name, path in zip(names[:n_nace_nodes], paths[:n_nace_nodes])],
    hovertemplate='%{text}<extra></extra>'
))

# 4. Ajouter les libellés (étoiles)
fig.add_trace(go.Scatter(
    x=X[n_nace_nodes:], 
    y=Y[n_nace_nodes:],
    mode='markers',
    name='Libellés',
    marker=dict(
        size=15,
        color='red',
        symbol='star',
        line=dict(width=1, color='darkred')
    ),
    text=[f"<b>{name}</b><br><br>{path}" for name, path in zip(names[n_nace_nodes:], paths[n_nace_nodes:])],
    hovertemplate='%{text}<extra></extra>'
))

accuracy = len(correct_predictions) / len(knn_edges) * 100
fig.update_layout(
    title=f"k-NN vs Ground Truth (Accuracy: {accuracy:.1f}%), DR method = {REDUC}",
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    width=1400,
    height=900,
    hovermode='closest',
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

fig.show()

# %%
fig.write_html("umap_visualization.html")

# %% Visualisation comparative des 4 méthodes avec lignes vertes
def create_comparison_plot(embeddings, nace_embeddings, label_embeddings, 
                          n_nace_nodes, names, paths, knn_edges, label_to_code_idx):
    """
    Crée un plot avec les 4 méthodes de réduction dimensionnelle côte à côte.
    Lignes vertes = k-NN correct, rouges = ground truth si erreur, bleues = k-NN incorrect
    """
    methods = ["umap", "pacmap", "tsne", "pca"]
    
    # Identifier les prédictions correctes
    correct_predictions = set()
    for label_idx, nace_idx, dist in knn_edges:
        if label_to_code_idx.get(label_idx) == nace_idx:
            correct_predictions.add((label_idx, nace_idx))
    
    n_correct = len(correct_predictions)
    n_total = len(label_to_code_idx)
    accuracy = n_correct / n_total * 100
    
    print(f"\n✓ Prédictions correctes: {n_correct}/{n_total} = {accuracy:.1f}%\n")
    
    # Calculer les coordonnées pour chaque méthode
    all_coords = {}
    method_names = {}
    
    for method in methods:
        coords, method_name = reduce_dimensions(embeddings, method=method)
        all_coords[method] = coords
        method_names[method] = method_name
        print(f"✓ {method_name} terminé")
    
    # Créer subplot 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[method_names[m] for m in methods],
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # Mapping méthode -> position subplot
    positions = {
        "umap": (1, 1),
        "pacmap": (1, 2),
        "tsne": (2, 1),
        "pca": (2, 2)
    }
    
    for method in methods:
        coords = all_coords[method]
        X, Y = coords.T
        row, col = positions[method]
        
        # 1. Lignes k-NN INCORRECTES (bleu)
        for label_idx, nace_idx, dist in knn_edges:
            if (label_idx, nace_idx) not in correct_predictions:
                alpha = max(0.3, 1 - dist/2)
                
                fig.add_trace(go.Scatter(
                    x=[X[label_idx], X[nace_idx]],
                    y=[Y[label_idx], Y[nace_idx]],
                    mode='lines',
                    line=dict(color=f'rgba(100, 150, 255, {alpha})', width=1.5, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row, col=col)
        
        # 2. Lignes ground truth pour cas INCORRECTS uniquement (rouge)
        for label_idx, code_idx in label_to_code_idx.items():
            # Vérifier si k-NN a trouvé le bon code
            is_correct = any((label_idx, nace_idx) in correct_predictions 
                            for _, nace_idx, _ in knn_edges if _ == label_idx)
            
            if not is_correct:
                fig.add_trace(go.Scatter(
                    x=[X[label_idx], X[code_idx]],
                    y=[Y[label_idx], Y[code_idx]],
                    mode='lines',
                    line=dict(color='rgba(255, 100, 100, 0.8)', width=2, dash='solid'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row, col=col)
        
        # 3. Lignes k-NN CORRECTES (vert épais) - PAR-DESSUS
        for label_idx, nace_idx, dist in knn_edges:
            if (label_idx, nace_idx) in correct_predictions:
                fig.add_trace(go.Scatter(
                    x=[X[label_idx], X[nace_idx]],
                    y=[Y[label_idx], Y[nace_idx]],
                    mode='lines',
                    line=dict(color='rgba(50, 205, 50, 0.9)', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row, col=col)
        
        # 4. Points NACE (cercles)
        show_legend_nace = (row == 1 and col == 1)
        fig.add_trace(go.Scatter(
            x=X[:n_nace_nodes], 
            y=Y[:n_nace_nodes],
            mode='markers',
            name='Codes NACE',
            legendgroup='nace',
            showlegend=show_legend_nace,
            marker=dict(
                size=6,
                color=np.arange(n_nace_nodes),
                colorscale='Viridis',
                showscale=False,
                line=dict(width=0.3, color='white'),
                symbol='circle'
            ),
            text=[f"<b>{name}</b><br>{path}" for name, path in zip(names[:n_nace_nodes], paths[:n_nace_nodes])],
            hovertemplate='%{text}<extra></extra>'
        ), row=row, col=col)
        
        # 5. Points libellés (étoiles)
        show_legend_label = (row == 1 and col == 1)
        fig.add_trace(go.Scatter(
            x=X[n_nace_nodes:], 
            y=Y[n_nace_nodes:],
            mode='markers',
            name='Libellés',
            legendgroup='labels',
            showlegend=show_legend_label,
            marker=dict(
                size=10,
                color='red',
                symbol='star',
                line=dict(width=0.8, color='darkred')
            ),
            text=[f"<b>{name}</b><br>{path}" for name, path in zip(names[n_nace_nodes:], paths[n_nace_nodes:])],
            hovertemplate='%{text}<extra></extra>'
        ), row=row, col=col)
        
        # Mise en forme des axes
        fig.update_xaxes(
            showgrid=True, 
            gridcolor='lightgray',
            showticklabels=False,
            title_text="Dim 1" if row == 2 else "",
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='lightgray',
            showticklabels=False,
            title_text="Dim 2" if col == 1 else "",
            row=row, col=col
        )
    
    # Ajouter légende pour les types de lignes (seulement visible sur premier subplot)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(50, 205, 50, 0.9)', width=3),
        name='k-NN ✓ Correct',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(100, 150, 255, 0.6)', width=1.5, dash='dot'),
        name='k-NN ✗ Incorrect',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(255, 100, 100, 0.8)', width=2),
        name='Ground Truth (si erreur)',
        showlegend=True
    ), row=1, col=1)
    
    fig.update_layout(
        title_text=(f"Comparaison des méthodes de réduction dimensionnelle (k-NN={K_NN})<br>" +
                   f"<sub>Accuracy: {n_correct}/{n_total} = {accuracy:.1f}% | " +
                   f"Vert=Correct, Rouge=Vraie cible si erreur, Bleu=Prédiction incorrecte</sub>"),
        title_font_size=16,
        height=1000,
        width=1600,
        hovermode='closest',
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


# Générer le plot comparatif
fig_comparison = create_comparison_plot(
    embeddings=embeddings,
    nace_embeddings=nace_embeddings,
    label_embeddings=label_embeddings,
    n_nace_nodes=n_nace_nodes,
    names=names,
    paths=paths,
    knn_edges=knn_edges,
    label_to_code_idx=label_to_code_idx
)

fig_comparison.show()


# %%
fig_comparison.write_html("comparison_all_methods.html")

# %%

from src.config import neo4j_config
from src.navigator.navigator import Navigator
from src.utils.logging import configure_logging
from src.utils.parser import parse_args
navigator = Navigator(neo4j_config)
tools = navigator.get_tools()

# %%
print(tools)
# %%
navigator._cached_get_parent(navigator.current_code)
# %%
navigator.current_code='01'
# 
# %%
tools[2]
