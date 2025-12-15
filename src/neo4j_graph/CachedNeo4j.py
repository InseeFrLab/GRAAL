from langchain_neo4j import Neo4jGraph
import hashlib
import json
from typing import Dict, Optional, List, OrderedDict
import logging 

logger = logging.getLogger(__name__)


class MultitonMeta(type):
    """Metaclass to create a multiton (one instance per graph)"""
    _instances: Dict[str, 'CachedNeo4jQuery'] = {}
    
    def __call__(cls, graph: Neo4jGraph, max_size: int = 1000):
        # Create an unique key for each graph
        key = cls._make_key(graph, max_size)
        
        if key not in cls._instances:
            instance = super().__call__(graph, max_size)
            cls._instances[key] = instance
            logger.info(f"Created new CachedNeo4jQuery instance (max_size={max_size})")
        else:
            logger.debug(f"Reusing existing CachedNeo4jQuery instance")
        
        return cls._instances[key]
    
    @staticmethod
    def _make_key(graph: Neo4jGraph, max_size: int) -> str:
        """
        Create an unique key for identifying a graph
        """
        graph_id = id(graph)
        key = f"graph_{graph_id}_size_{max_size}"
        return hashlib.md5(key.encode()).hexdigest()


class CachedNeo4jQuery(metaclass=MultitonMeta):
    """Cache wrapper for Neo4j queries with LRU eviction strategy.
    Due to multiton pattern, __init__ is called every time,
    but initialization only happens once per unique graph."""
    
    def __init__(self, graph: Neo4jGraph, cache_size: int):
        # Avoid re-initialization (multiton)
        if hasattr(self, '_initialized'):
            return
        
        self.graph = graph
        self.cache_size = cache_size
        self._query_cache = OrderedDict[str, List[Dict]] = OrderedDict()
    
    def _generate_cache_key(self, query: str, params: Dict) -> str:
        """Generate a unique cache key for a query and its parameters"""
        param_str = json.dumps(params, sort_keys=True)
        content = f"{query}::{param_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def query(self, query_str: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute query with caching"""
        params = params or {}
        cache_key = self._generate_cache_key(query_str, params)
        
        # Check cache first
        if cache_key in self._query_cache:
            logger.debug(f"Cache hit for query: {query_str[:50]}...")
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]
        
        # Execute query if not in cache
        logger.debug(f"Cache miss, executing query: {query_str[:50]}...")
        result = self.graph.query(query_str, params=params)
        
        # Store in cache (with size limit)
        if len(self._query_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            evicted_key = next(iter(self._query_cache))
            del self._query_cache[evicted_key]
            logger.debug("Cache full, removed oldest entry")
        
        self._query_cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")
