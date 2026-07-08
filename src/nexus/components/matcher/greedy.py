
from typing import List
import numpy as np
from nexus.core.interfaces.matcher import Matcher
from nexus.utils.datastructures import UnionFind, MatchingResult
from nexus.utils.timing import timed


class GreedyMatcher(Matcher):
    def __init__(self, similarity_threshold: float = 0.01):
        """Greedy matcher using similarity functions from DataModel.
        
        Args:
            similarity_threshold: Minimum similarity for filtering candidates
        """
        self.similarity_threshold = similarity_threshold


    def _filter_and_sort(self, candidates, data: "DataModel"):
        """Filter candidates with zero similarity and sort by confidence descending"""
        valid_candidates = []
        for elems, dist in candidates:
            sim = data.compute_similarity(elems)
            if sim > 0.0:
                valid_candidates.append((elems, sim))
        
        # Sort by similarity descending (highest similarity first)
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_candidates
    

    @timed()
    def __call__(self, candidates: List, data: "DataModel"):
        """Greedy matching phase using DataModel's similarity methods"""

        # Initialize Union-Find structure
        uf = UnionFind(len(data.df))
        
        # Filter and sort candidates by similarity
        candidates = self._filter_and_sort(candidates, data)
        
        # Merging step
        for ((e1_idx, e2_idx), dist) in candidates:
            e1_parent_idx = uf.find(e1_idx)
            e2_parent_idx = uf.find(e2_idx)
            
            if e1_parent_idx == e2_parent_idx:
                continue
            
            comp1 = set(uf.get_component(e1_parent_idx))
            comp2 = set(uf.get_component(e2_parent_idx))

            # Check if merger is valid (no document conflicts) and similarity increases
            if (self._is_valid(comp1, comp2, data) and 
                data.shouldMatch(comp1, comp2, threshold=self.similarity_threshold)):
                uf.union(e1_parent_idx, e2_parent_idx)

        # Extract final components and compute their similarities
        components = uf.get_all_components().values()
        components_w_sim = []
        for c in components:
            sim = data.compute_similarity(list(c))
            if sim is None or (isinstance(sim, float) and np.isnan(sim)):
                sim = 0.0
            components_w_sim.append((c, sim))

        return MatchingResult(
            matches=components_w_sim,
            timing={}
        )
    

    def _is_valid(self, match1: set, match2: set, data: "DataModel") -> bool:
        """Check if two matches can be merged (no model/document conflicts)"""
        return len(data.get_documents_from_index(match1) & data.get_documents_from_index(match2)) == 0