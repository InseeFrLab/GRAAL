'''
TODO: 
- Représentations des embeddings par t-SNE, UMAP, PCA
- Test sur la Coicop
- Shuffle des bases
- Evaluation sur différents agents
'''
# T-SNE

# Récupération des embeddings

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import logging

from src.config import neo4j_config
from src.neo4j_graph.graph import Graph

logger = logging.getLogger(__name__)


graph = Graph(neo4j_config)
nb_embeddings = 2
sample_embeddings = graph.graph.query(
    """MATCH (n) 
    WHERE n.embedding IS NOT NULL
    RETURN n.embedding as embedding
    LIMIT $nb_embeddings
    """, params={"nb_embeddings": nb_embeddings}
)

embeddings_list = [record["embedding"] for record in sample_embeddings]
print(embeddings_list)
embeddings_array = np.array(embeddings_list)

reducer = umap.UMAP()
embed_reduced = reducer.fit_transform(sample_embeddings)
logger.info(f"Dimension of the reduction: {embed_reduced.shape}")

