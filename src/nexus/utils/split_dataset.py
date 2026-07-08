"""Utilities to generate train/eval temporary dataset splits.

Provides a helper that reads a CSV, randomly splits rows into two parts,
writes each part to a temporary CSV file on disk, and returns the two
file paths.
"""
import tempfile
from typing import Tuple
import pandas as pd
import numpy as np

def generate_test_eval_split(
    df: pd.DataFrame,
    fraction: float,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into two sets by assigning whole groups, so that the first set
    is as close as possible to fraction * len(df) rows.
    """
    rng = np.random.RandomState(random_state)
    group_col = df.columns[1]
    groups = df[group_col].unique()
    # groups = groups.to_numpy() 
    rng.shuffle(groups)

    group_sizes = df.groupby(group_col).size().to_dict()
    total_rows = len(df)
    target_rows = int(total_rows * fraction)

    split1_groups = []
    split1_rows = 0

    for group in groups:
        group_row_count = group_sizes[group]
        if split1_rows + group_row_count <= target_rows or not split1_groups:
            split1_groups.append(group)
            split1_rows += group_row_count
        else:
            break

    split1 = df[df[group_col].isin(split1_groups)].copy()
    split2 = df[~df[group_col].isin(split1_groups)].copy()
    
    split1_file = tempfile.NamedTemporaryFile(delete=True, suffix=".csv", mode="w")
    split2_file = tempfile.NamedTemporaryFile(delete=True, suffix=".csv", mode="w")
    split1.to_csv(split1_file.name, index=False, header=False)
    split2.to_csv(split2_file.name, index=False, header=False)

    return split1_file, split2_file