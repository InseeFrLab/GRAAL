# %% Imports
import os
import numpy as np
import plotly.graph_objects as go
import umap
from src.config import neo4j_config
from src.neo4j_graph.graph import Graph
# Factorize the code to embed
# from src.neo4j_graph.graph_builder.utils.embed_manager import get_embedding_model
from langchain_openai import OpenAIEmbeddings

# %% Récupération des données
graph = Graph(neo4j_config)

query = """
MATCH path = (root)-[*]->(n)
WHERE n.LEVEL = 5
  AND n.embedding IS NOT NULL
  AND root.LEVEL = 0
RETURN n.embedding as embedding,
       n.NAME as name,
       [node IN nodes(path) | node.CODE] as path_codes,
       [node IN nodes(path) | node.LEVEL] as path_levels
"""

results = graph.graph.query(query)

embeddings = []
names = []
paths = []

for record in results:
    embeddings.append(record["embedding"])
    names.append(record["name"])
    
    path_str = " → ".join([
        name for lvl, name in zip(record["path_levels"], record["path_codes"])
    ])
    paths.append(path_str)


print(f"Nœuds récupérés: {len(names)}")
print(embeddings)


# %%
# Add a query in the embedding space

queries = ["Je vends des croissants", "Livreur de taxi", "Coiffeur"]
emb_model = OpenAIEmbeddings(
        model=os.environ['EMBEDDING_MODEL'],
        openai_api_base=os.environ['URL_EMBEDDING_API'],
        openai_api_key="EMPTY",
        tiktoken_enabled=False,
    )
for i, query in iter(queries): 
    query_emb = emb_model.embed_query(query)
    embeddings.append(query_emb)
    names.append(f"Query {i}")
    paths.append(query)


# %% UMAP
reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.1)
embeddings = np.array(embeddings)
coords = reducer.fit_transform(embeddings)
X, Y = coords.T

# %% Visualisation interactive
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=X, y=Y,
    mode='markers',
    marker=dict(
        size=10,
        color=np.arange(len(X)),  # Couleur par index
        colorscale='Viridis',
        showscale=True,
        line=dict(width=0.5, color='white')
    ),
    text=[f"<b>{name}</b><br><br>{path}" for name, path in zip(names, paths)],
    hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(
    title="Nœuds de niveau 5",
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    width=1200,
    height=800,
    hovermode='closest',
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)

fig.show()

# %%
fig.write_html("niveau5_simple.html")
# %%
