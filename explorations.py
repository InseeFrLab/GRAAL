# %% Imports
import os
import numpy as np
import plotly.graph_objects as go
from langchain_openai import OpenAIEmbeddings
import polars as pl
import umap
import asyncio

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
graph = Graph(neo4j_config)

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


print(f"Nœuds récupérés: {len(names)}")

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

path = "projet-ape/data/08112022_27102024/naf2025/split/df_train.parquet"
columns = ["libelle", "nace2025"]

codes = sample_codes(
    fs=fs,
    population_path=path,
    code_column=columns, 
    n_codes=N_CODES)

labels, target_codes = zip(*codes)

emb_model = OpenAIEmbeddings(
        model=os.environ['EMBEDDING_MODEL'],
        openai_api_base=os.environ['URL_EMBEDDING_API'],
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )

labels_embeddings = emb_model.embed_documents(list(labels))

label_to_code_idx = {}

for i, (label, label_emb, target_code) in enumerate(zip(labels, labels_embeddings, target_codes)):
    embeddings.append(label_emb)
    names.append(label[:50])
    paths.append(f"Libellé -> Code cible: {target_code}")

    label_idx = n_nace_nodes + i
    if target_code in codes_dict: 
        label_to_code_idx[label_idx] = codes_dict[target_code]
        
# %%
print(label_to_code_idx)

# %% UMAP
reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.1)
embeddings = np.array(embeddings)
coords = reducer.fit_transform(embeddings)
X, Y = coords.T




# %% Visualisation interactive
fig = go.Figure()


# 1. Ajouter les lignes de connexion AVANT les points
for label_idx, code_idx in label_to_code_idx.items():
    fig.add_trace(go.Scatter(
        x=[X[label_idx], X[code_idx]],
        y=[Y[label_idx], Y[code_idx]],
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.8)', width=3, dash='solid'),
        showlegend=False,
        hoverinfo='skip'
    ))

# 2. Ajouter les nœuds NACE (cercles)
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

# 3. Ajouter les libellés (étoiles)
fig.add_trace(go.Scatter(
    x=X[n_nace_nodes:], 
    y=Y[n_nace_nodes:],
    mode='markers',
    name='Libellés',
    marker=dict(
        size=15,
        color='red',
        symbol='star',  # ou 'diamond', 'square', 'cross', 'x', 'triangle-up'
        line=dict(width=1, color='darkred')
    ),
    text=[f"<b>{name}</b><br><br>{path}" for name, path in zip(names[n_nace_nodes:], paths[n_nace_nodes:])],
    hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(
    title="Nœuds NACE niveau 5 et libellés échantillonnés",
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

# %%
result = await classify_navigator(labels[0])


# %%
print(result)
# %%
