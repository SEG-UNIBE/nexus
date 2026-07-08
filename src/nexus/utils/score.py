import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple
from nexus.core.interfaces.data_model import DataModel

def pairs(clusters):
    """Generate all unique pairs from a list of clusters (lists or sets of identifiers)."""
    all_pairs = set()
    for cluster in clusters:
        elements = list(cluster)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                e1, e2 = elements[i], elements[j]
                if e1 == e2: 
                    continue
                pair = tuple(sorted((e1, e2)))
                all_pairs.add(pair)
    return all_pairs


def group_cohesion(G: np.ndarray, D_max: float) -> float:
    """
    Calculates the normalized Group Cohesion (Tightness) T(G).
    T(G) = max(0, 1 - D(G) / D_max)

    Args:
        G (np.ndarray): The group of vectors. Shape (N_i, D).
        D_max (float): The maximum possible discrepancy (e.g., total dataset variance).

    Returns:
        float: The cohesion score between 0 and 1.
    """
    if G.shape[0] == 0:
        return 0.0
    if D_max <= 1e-9:  # Avoid division by zero if D_max is near zero
        return 1.0

    # Calculate the centroid (mean vector)
    centroid = np.mean(G, axis=0)

    # Calculate Discrepancy D(G) (Within-Cluster Sum of Squares - WCSS)
    # ||v - c||^2 is the sum of squared differences
    discrepancy = np.sum(np.sum((G - centroid) ** 2, axis=1))
    
    # Normalize and return cohesion score
    cohesion_score = 1.0 - (discrepancy / D_max)
    return np.clip(cohesion_score, 0.0, 1.0)


def group_size_punishment(N: int, alpha: float) -> float:
    """
    Calculates the Group Size Punishment (P(G)) factor.
    
    Strategy: 1 - 1 / (N ^ alpha)
    - Penalizes singletons (score 0).
    - Rewards merging (score approaches 1 as N grows).

    Args:
        N (int): Size of the current group.
        alpha (float): Saturation speed (default 1.0).

    Returns:
        float: The size punishment score between 0 and 1.
    """
    if N <= 1:
        return 0.0
    
    return 1.0 - (1.0 / (float(N) ** alpha))


def avg_pairwise_cosine_similarity(G: np.ndarray) -> float:
    """
    Computes the average pairwise cosine similarity for all vectors in G.
    Args:
        G (np.ndarray): Shape (N, D)
    Returns:
        float: Average pairwise cosine similarity
    """
    if G.shape[0] < 2:
        return 1.0  # Only one vector, trivially similar

    # Normalize each vector to unit length
    G_norm = G / np.linalg.norm(G, axis=1, keepdims=True)
    # Compute cosine similarity matrix
    S = np.dot(G_norm, G_norm.T)
    # Exclude diagonal (self-similarity)
    N = G.shape[0]
    mask = ~np.eye(N, dtype=bool)
    avg_cosine = S[mask].mean()
    return avg_cosine


def total_cosine_sim_quality(groups: list[np.ndarray], X: np.ndarray, alpha: float = 2.0, cosine_reduction="wcss") -> float:
    """
    Computes the total partition quality metric S_total(X) as a weighted average.

    S_total(X) = sum(|G_i| / N_total) * S_cosine(G_i) * P(G_i)
    
    where P(G_i) = 1 - (1 / |G_i|^alpha)

    """
    N_total = X.shape[0]
    if N_total == 0:
        return 0.0

    total_score = 0.0

    for G in groups:
        if cosine_reduction == "ccc":
            c = np.mean(G, axis=0)
            # Fix: Normalize centroid to get valid cosine similarity [-1, 1]
            c_norm_val = np.linalg.norm(c)
            if c_norm_val > 1e-9:
                c = c / c_norm_val
            
            # Handle potential zero vectors to avoid NaN
            norms = np.linalg.norm(G, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            G_norm = G / norms
            
            T_G = np.mean(np.dot(G_norm, c.T))
        else: # wcss
            T_G = avg_pairwise_cosine_similarity(G)

        N = G.shape[0]
        P_G = group_size_punishment(N, alpha)

        weight = N / N_total
        total_score += weight * T_G * P_G

    return total_score


def total_partition_quality(groups: list[np.ndarray], X: np.ndarray, alpha: float = 2.0) -> float:
    """
    Computes the Total Partition Quality Metric S_total(X) as a weighted average.

    S_total(X) = sum( (|G_i| / N_total) * S(G_i) )
    where S(G_i) = T(G_i) * P(G_i)
    and P(G_i) = 1 - (1 / |G_i|^alpha)

    Args:
        groups (list[np.ndarray]): A list of numpy arrays, where each array is a group (cluster).
        X (np.ndarray): The original full dataset.
        alpha (float): Exponent for the size punishment factor (default is 2.0).

    Returns:
        float: The overall quality score for the partitioning (0 to 1).
    """
    N_total = X.shape[0]
    if N_total == 0:
        return 0.0

    # 1. Calculate D_max (Maximum possible Discrepancy)
    # This is typically the Total Sum of Squares (TSS) for the entire dataset X.
    # We use this to normalize all group discrepancies.
    X_centroid = np.mean(X, axis=0)
    D_max = np.sum((X - X_centroid) ** 2)

    # Handle the case where all vectors are identical (D_max is 0)
    if D_max < 1e-9:
        # If D_max is near zero, all partitions are perfect (T(G) is 1), so score is 1.0 if not empty.
        return 1.0 if N_total > 0 else 0.0

    total_score = 0.0
    
    for G in groups:
        N = G.shape[0]
        
        # 2. Calculate T(G) - Cohesion Score
        T_G = group_cohesion(G, D_max)
        
        # 3. Calculate P(G) - Size Punishment Factor
        P_G = group_size_punishment(N, alpha)
        
        # 4. Calculate S(G) - Single Group Quality Score
        S_G = T_G * P_G
        
        # 5. Add the weighted contribution to the total score
        weight = N / N_total
        total_score += weight * S_G
        
    return total_score


def total_rubin_and_chechik_score(matches: list[Tuple], data_model: DataModel) -> float:
    """Compute the Rubin & Chechik (2013) weighted score for a set of matches.

    Args:
        matches: list of tuples (members_iterable, reported_score?) where members_iterable
                 contains dataframe/global indices for the group.
        data_model: a DataModel instance providing element attribute access (preferably
                    via `attr_cache` or `_get_attr_from_idx`).

    Returns:
        float: total Rubin & Chechik score (sum over calculated tuple weights).

    Notes:
    - We ignore any externally reported per-match score and recompute weights
      using `calculate_tuple_weight` based on element properties extracted from
      the `data_model`.
    """
    # Normalize matches into list of integer index lists
    matches_list: list[list[int]] = []
    for item in matches:
        matches_list.append(item[0])

    # Build element_properties mapping: idx -> set(properties)
    element_properties: Dict[int, Set[str]] = {}

    # Prefer an existing attr_cache on the data model
    if hasattr(data_model, 'attr_cache') and isinstance(getattr(data_model, 'attr_cache'), dict):
        for k, v in data_model.attr_cache.items():
            try:
                ik = int(k)
            except Exception:
                ik = k
            if v is None:
                element_properties[ik] = set()
            else:
                try:
                    element_properties[ik] = set(v)
                except Exception:
                    # fallback: try iterating
                    element_properties[ik] = set(list(v))
    else:
        # Fallback: call `_get_attr_from_idx` for any index present in matches
        seen = set()
        for grp in matches_list:
            for idx in grp:
                if idx in seen:
                    continue
                seen.add(idx)
                props = set()
                p = data_model._get_attr_from_idx(int(idx))
                props = set(p) if p is not None else set()

                element_properties[int(idx)] = props

    # Compute weights per match and sum
    total_score = 0.0
    for grp in matches_list:
        w = calculate_tuple_weight(grp, element_properties, data_model.n_docs)
        total_score += w

    return float(total_score)

def calculate_tuple_weight(match_indices: List[int], 
                           element_properties: Dict[int, Set[str]], 
                           n_models: int) -> float:
    """
    Calculates the weight (score) of a single merged tuple based on the 
    N-Way Model Merging (2013) metric.

    Args:
        match_indices: A list of indices representing the elements in the match/tuple.
        element_properties: A dictionary mapping element index -> set of property strings.
                            Example: {1: {'id', 'office'}, 2: {'name', 'office'}}
        n_models: The total number of input models (N).

    Returns:
        The calculated weight (float) between 0.0 and 1.0.
    """
    # 1. Validate input
    if not match_indices or n_models == 0:
        return 0.0
    
    # Get the properties for each element in the tuple
    # Filter out any indices that might not exist in the properties map
    tuple_props_list = [element_properties[idx] for idx in match_indices if idx in element_properties]
    
    if not tuple_props_list:
        return 0.0

    # 2. Validity Check (Optional but recommended by the paper):
    # The paper implies a tuple is invalid (weight 0) if disjoint sets of elements exist
    # (i.e., every element must share at least one property with another element in the tuple).
    # This simple check ensures weight is 0 if no properties are shared at all.
    # For stricter validity (connectivity), a graph traversal would be needed.
    
    # 3. Calculate Property Distribution
    # Count how many elements have each property
    prop_counts = Counter()
    for props in tuple_props_list:
        prop_counts.update(props)
        
    total_distinct_properties = len(prop_counts)
    
    if total_distinct_properties == 0:
        return 0.0

    # 4. Calculate Numerator
    # Sum of (count^2) for all properties that appear in > 1 element.
    # Properties appearing in only 1 element contribute 0.
    numerator = sum(count**2 for count in prop_counts.values() if count > 1)

    # 5. Calculate Denominator
    # Normalized by N^2 * Total Distinct Properties
    denominator = (n_models ** 2) * total_distinct_properties

    return numerator / denominator

def score_merged_matches(matches_list: List[List[int]], 
                         element_properties: Dict[int, Set[str]], 
                         n_models: int) -> List[Tuple[List[int], float]]:
    """
    Takes a list of matches (indices) and returns a list of (match, score) tuples.
    Also calculates the total score of the solution.
    """
    scored_matches = []
    total_score = 0.0
    
    for match in matches_list:
        score = calculate_tuple_weight(match, element_properties, n_models)
        scored_matches.append((match, score))
        total_score += score
        
    return scored_matches, total_score
