
from __future__ import annotations
from typing import List, Tuple, Set
from abc import ABC, abstractmethod
from nexus.utils.timing import timed

import pandas as pd
import numpy as np

# TODO: add BCubed metric

class DataModel(ABC):
    vector_embeddings: np.ndarray
    embedding_fn: callable
    
    def __init__(self, embedding_strategy: str = 'lh', **embedding_params):
        """
        Args:
            embedding_strategy: Which vectorization method to use
            **embedding_params: Additional params for the embedding (e.g., reduce_dim)
        """
        self.embedding_fn: callable = None
        self.vector_embeddings = None
        self.embedding_strategy = embedding_strategy
        self.embedding_params = embedding_params
        self.embedding_dim = 0

        self.n_docs = 0
        self.total_elements = 0
    
    @abstractmethod
    @timed()
    def load(self) -> 'DataModel':
        """Load data from source (file, DB, etc.) into internal representation (e.g., df)"""
        raise NotImplementedError("Subclasses must implement load.")


    @abstractmethod
    def get_doc_id(self, idx):
        """Returns the document Id of the element index"""
        raise NotImplementedError("Subclasses must implement get_doc_id.")
    
    @timed()
    def embed(self) -> np.ndarray:
        """Compute vector embeddings from loaded data using the chosen strategy.
        
        Returns:
            np.ndarray: Vector embeddings of shape (n_items, embedding_dim)
        """
        if self.embedding_fn is None:
            raise ValueError(f"No embedding function provided")

        data_points = [self.__getitem__(i) for i in range(len(self))]
        self.vector_embeddings = self.embedding_fn(data_points, **self.embedding_params)

        return self.vector_embeddings
        
    def __iter__(self):
        """Iterate through dataset"""
        self.__index__ = 0
        for i in range(len(self)):
            yield self[i]
            self.__index__ += 1

    @abstractmethod
    def __getitem__(self, idx) -> object:
        """Get item i from the dataset"""
        raise NotImplementedError("Subclasses must implement __getitem__.")

    @abstractmethod
    def __len__(self):
        """Return the number of items in the dataset"""
        raise NotImplementedError("Subclasses must implement __len__.")

    @timed()
    def __call__(self) -> Tuple['DataModel', np.ndarray]:
        """Convenience: load + embed in one call.
        
        Returns:
            (self, vectors) where self has loaded data and vectors are embeddings
        """
        self.load()
        vectors = self.get_vector_embeddings()
        self.vector_embeddings = vectors
        return self, vectors
    
    
    @abstractmethod
    def get_documents_from_index(self, group: set) -> set:
        pass
    
    
    @abstractmethod
    def compute_similarity(self, indices: List[int]) -> float:
        """Compute similarity score for a group of elements.
        
        Args:
            indices: List of global element indices to compute similarity for

        Returns: 
            float: similarity score
        """
        pass
    
    @abstractmethod
    def shouldMatch(self, group1: Set[int], group2: Set[int], threshold: float = 0.01) -> bool:
        """Check if two groups should be merged based on similarity increase.
        
        Args:
            group1: Set of element indices in first group
            group2: Set of element indices in second group
            threshold: Minimum similarity threshold
            
        Returns:
            bool: True if groups should be merged
        """
        raise NotImplementedError("Subclasses must implement shouldMatch.")


    def get_total_elements(self) -> int:
        """ Returns the total number of data elements in all documents"""
        return self.total_elements
    
    def get_vector_embeddings(self) -> np.ndarray:
        if self.vector_embeddings is None:
            self.embed()
        
        return self.vector_embeddings
    

    @abstractmethod
    def get_document_group(self, doc_repr) -> List:
        pass
    
    def export_matches(self, matches: list, output_path: str) -> pd.DataFrame:
        """
        Subclasses should override this method to provide format-specific export logic.
        
        Args:
            matches: List of (set_of_indices, similarity) tuples from MatchingResult
            output_path: Path to save the CSV file
            
        Returns:
            DataFrame with the exported matches
        """
        raise NotImplementedError("Subclasses should implement export_matches()")
    

    @abstractmethod
    def compute_metrics(self, matches: list) -> Tuple[float]:
        """
        Computes accuracy, precision, recall and f1 metrics
        """
        pass

    @abstractmethod
    def compute_proxy(self, matches: list) -> float:
        """Compute proxy score for a set of matches
        
        Args:
            indices: List of global element indices to compute proxy for the matches

        Returns: 
            float: proxy score
        """
        pass