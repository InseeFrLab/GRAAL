# %% Imports minimalistes
import numpy as np
import plotly.graph_objects as go
import umap
from src.config import neo4j_config
from src.neo4j_graph.graph import Graph

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

embeddings = np.array(embeddings)
print(f"Nœuds récupérés: {len(names)}")

# %% UMAP
reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.1)
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
