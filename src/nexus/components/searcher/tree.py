from typing import List, Tuple, Dict, Set, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KDTree, BallTree

from nexus.core.interfaces.searcher import Searcher
from nexus.utils.timing import timed
from nexus.core.interfaces.data_model import DataModel


class TreeSearch(Searcher):
    def __init__(self, tree_method = "kdtree", k_neighbors = 5):
        super().__init__()
        self.tree_method = tree_method
        self.k_neighbors = k_neighbors

    @timed()
    def __call__(self, data: "DataModel", **kwargs) -> List[Tuple[List[int], float]]:
        """Find candidate matches using K-Nearest Neighbors (matches Java RaQuN)"""
        # Build kdtree structure
        embeddings = data.get_vector_embeddings()
        
        # Build mapping: vector tuple -> list of element indices (like Java's tree.get(vector))
        # This handles multiple elements at the same point
        vector_to_indices: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(len(embeddings)):
            vector_key = tuple(embeddings[i])
            vector_to_indices[vector_key].append(i)
        
        if self.tree_method == "kdtree":
            tree = KDTree(embeddings)
        elif self.tree_method == "balltree":
            tree = BallTree(embeddings)
        else:
            raise ValueError(f"Unknown kdtree method: {self.tree_method}")
        
        candidates_pairwise = []
        seen_pairs: Set[Tuple[int, int]] = set()
        
        # Find neighbors for each element (row)
        for i in range(len(data.df)):
            distances, indices = tree.query([embeddings[i]], k=min(self.k_neighbors + 1, len(data.df)))
            curr_doc_id = data.get_doc_id(i)
            
            # Process each neighboring point
            for dist, idx in zip(distances[0], indices[0]):
                # Get all elements at this neighboring point (like Java)
                neighbor_vector = tuple(embeddings[idx])
                elements_at_point = vector_to_indices[neighbor_vector]
                
                for neighbor_idx in elements_at_point:
                    # Skip self
                    if neighbor_idx == i:
                        continue
                    
                    neighbor_doc_id = data.get_doc_id(neighbor_idx)
                    
                    # Only add if from different models (validity constraint)
                    if curr_doc_id != neighbor_doc_id:
                        # Avoid duplicate pairs (normalize order)
                        pair = (min(i, neighbor_idx), max(i, neighbor_idx))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidates_pairwise.append(((i, neighbor_idx), dist))
        
        return candidates_pairwise
