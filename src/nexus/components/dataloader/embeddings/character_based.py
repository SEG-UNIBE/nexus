import numpy as np
from typing import List


def character_based_embedding(data: List[str]) -> np.ndarray:
    """
    CharacterBasedVectorization matching RaQuN's original Java implementation.
    
    Creates vectors with:
    - Dim 0: Number of properties (attributes)
    - Dim 1: Average property name length
    - Dims 2+: Character frequency counts (all unique characters in dataset)
    
    Two-pass approach:
    1. First pass: collect all unique characters from all properties
    2. Second pass: build vectors with dynamic dimensions
    """
    # check if element type is str
    if not all(isinstance(row, str) for row in data):
        raise 

    nr_elements = len(data)
    
    # First pass: collect all unique characters and parse attributes
    all_unique_chars = set()
    parsed_data = []

    for i, row in enumerate(data):
        attrs = row.split(";")
        all_unique_chars.update(row)
        parsed_data.append((attrs, row))
    
    # Build character-to-dimension mapping (starting at dim 2)
    sorted_chars = sorted(all_unique_chars)  # Sort for deterministic ordering
    char_to_dim = {c: i + 2 for i, c in enumerate(sorted_chars)}
    
    # Total dimensions: 2 (property count + avg length) + number of unique chars
    total_dims = 2 + len(sorted_chars)
    vectors = np.zeros((nr_elements, total_dims))
    
    # Second pass: build vectors
    for i, (all_attrs, row_text_lower) in enumerate(parsed_data):
        # Dim 0: Number of properties
        vectors[i, 0] = len(all_attrs)
        
        # Dim 1: Average property name length
        if all_attrs:
            vectors[i, 1] = np.mean([len(attr) for attr in all_attrs])
        else:
            vectors[i, 1] = 0.0
        
        # Dims 2+: Character frequency counts
        for char in row_text_lower:
            dim = char_to_dim[char]
            vectors[i, dim] += 1
    
    return vectors
