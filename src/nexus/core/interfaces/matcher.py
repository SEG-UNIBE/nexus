
from __future__ import annotations
from typing import TYPE_CHECKING, List
from abc import ABC, abstractmethod

class Matcher(ABC):
    """Matcher component for final matching"""
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, candidates: List, data: "DataModel"):
        """Execute matching on candidates"""
        pass

