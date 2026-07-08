
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

class UnionFind:
    def __init__(self, n):
        """Initialize Union-Find with n elements (0 to n-1)"""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_components = n
        self.components = {i: [i] for i in range(n)}

        
    def find(self, x):
        """Find root of element x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def connected(self, x, y):
        """Check if two elements are in the same component"""
        return self.find(x) == self.find(y)
    
    def component_size(self, x):
        """Get size of component containing x"""
        return self.size[self.find(x)]

    def union(self, x, y):
        """Union two elements by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
            self.components[root_y].extend(self.components[root_x])
            del self.components[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.components[root_x].extend(self.components[root_y])
            del self.components[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
            self.components[root_x].extend(self.components[root_y])
            del self.components[root_y]
        
        self.num_components -= 1
        return True
    
    def get_component(self, x):
        """Get component containing x - O(1) after find()"""
        root = self.find(x)
        return self.components[root]
    
    def get_all_components(self):
        """Get all components - O(1)"""
        return self.components


@dataclass
class MatchingResult:
    """Matching result"""
    matches: List[Tuple[List, int]]
    timing: Dict[str, float]