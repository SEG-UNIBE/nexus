
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any
from abc import ABC, abstractmethod

class SamplingStrategy(ABC):
    @abstractmethod
    def init(self, param_registry: "ParameterRegistry") -> None:
        """Called once at the start"""
        pass
    
    @abstractmethod
    def next_batch(self, param_registry: "ParameterRegistry") -> List[Dict]:
        """Returns next config or None when done"""
        pass
    
    @abstractmethod
    def update_batch(self, results: List[Dict[str, Any]]) -> None:
        """Receive feedback from completed job"""
        pass
    
    @abstractmethod
    def has_more(self) -> bool:
        """Check if more configs are available"""
        pass
