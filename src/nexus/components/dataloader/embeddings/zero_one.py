import numpy as np
from typing import List

def zero_one_embedding(data: List[str]) -> np.ndarray:
    all_row_attrs = []
    all_attributes = set()
    for row in data:
        attrs = [attr.strip() for attr in row.split(';') if attr.strip()]
        all_row_attrs.append(attrs)
        all_attributes.update(attrs)

    attr_to_index = {attr: i for i, attr in enumerate(sorted(all_attributes))}

    vectors = np.zeros((len(all_row_attrs), len(all_attributes)))
    for i, attrs in enumerate(all_row_attrs):
        for attr in attrs:
            if attr in attr_to_index:  # Safety check
                vectors[i, attr_to_index[attr]] = 1

    return vectors
