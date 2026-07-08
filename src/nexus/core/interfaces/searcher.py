
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

class Searcher(ABC):
    """Search component for finding candidates"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, data: "DataModel", **kwargs) -> List[Tuple[List[int], float]]:
        pass