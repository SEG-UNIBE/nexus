from typing import Dict, Any, Tuple, List

def get_pareto_front(results, t_metric="total_time", s_metric="proxy") -> Dict:
    """
    Computes the pareto front from a given Ray Tune results list,
    and returns the configuration with the best proxy score on the pareto front.
    Keywords to sort: "proxy", "total_time"
    """
    all_pts = []
    for r in results:
        time = r.metrics.get(t_metric)
        score = r.metrics.get(s_metric)
        if time is not None and score is not None:
            all_pts.append((time, score, r.config, r.metrics))

    if not all_pts:
        return {}

    # 1. Sort by time (ascending), then by score (descending)
    sorted_results = sorted(all_pts, key=lambda x: (x[0], -x[1]))
    
    pareto_front = []
    max_score_so_far = -float('inf')
    
    # 2. Iterate and Filter
    for time, score, config, metrics in sorted_results:
        if score > max_score_so_far:
            pareto_front.append((time, score, config, metrics))
            max_score_so_far = score
    
    return pareto_front

def get_top_n_secant_dist_points(pareto_front: List[Tuple], n: int = 1) -> List[Tuple]:
    """ 
    input: 
        pareto_front: List[Tuple] - a list of (time, score, config, metrics) 
        n: int - the number of "elbow" points to return
    output: 
        List[Tuple] - a list of the top N points, sorted by their secant distance using 
        the normalized Kneedle method.
    """ 
    if not pareto_front:
        return []

    # If the front has fewer or exactly n points, just return what's available
    if len(pareto_front) <= n:
        return sorted(pareto_front, key=lambda x: x[0])

    sorted_front = sorted(pareto_front, key=lambda x: x[0])

    min_time = sorted_front[0][0]
    max_time = sorted_front[-1][0]
    
    # Safely find min and max scores across the whole front
    scores = [pt[1] for pt in sorted_front]
    min_score = min(scores)
    max_score = max(scores)

    # Fallback if there is no variance to normalize against
    if max_time == min_time or max_score == min_score:
        return sorted_front[:n]

    # Calculate distances for all points
    points_with_dist = []
    for pt in sorted_front:
        time, score, config, metrics = pt
        
        norm_time = (time - min_time) / (max_time - min_time)
        norm_score = (score - min_score) / (max_score - min_score)
        
        dist = abs(norm_time - norm_score) / (2 ** 0.5)
        points_with_dist.append((dist, pt))
        
    # Sort by distance in descending order to get the largest "elbows" first
    points_with_dist.sort(key=lambda x: x[0], reverse=True)

    # Extract and return only the original point tuples (dropping the distance)
    return [pt for dist, pt in points_with_dist[:n]]


def get_top_n_utopia_points(pareto_front: List[Tuple], n: int = 1) -> List[Tuple]:
    """ 
    input: 
        pareto_front: List[Tuple] - a list of (time, score, config, metrics) 
        n: int - the number of top points to return (defaults to 1)
    output: 
        List[Tuple] - returning a list of the n models closest to the normalized Utopia point.
    """
    if not pareto_front:
        return []

    # Safely bound n to the length of the pareto front
    n = min(n, len(pareto_front))

    min_time = min(pt[0] for pt in pareto_front)
    max_time = max(pt[0] for pt in pareto_front)
    
    min_score = min(pt[1] for pt in pareto_front)
    max_score = max(pt[1] for pt in pareto_front)

    # Edge case: no variance in time or score
    if max_time == min_time or max_score == min_score:
        return pareto_front[:n]

    # List to hold tuples of (distance, original_point)
    points_with_distances = []

    for pt in pareto_front:
        time, score = pt[0], pt[1]
        
        norm_time = (time - min_time) / (max_time - min_time)
        norm_score = (score - min_score) / (max_score - min_score)
        
        # Utopia point is (0, 1): 0 for normalized time (minimized), 1 for normalized score (maximized)
        dist = ((norm_time - 0) ** 2 + (norm_score - 1) ** 2) ** 0.5
        
        points_with_distances.append((dist, pt))

    # Sort the points by distance in ascending order (closest to Utopia point first)
    points_with_distances.sort(key=lambda x: x[0])

    # Extract the original points from the sorted list and return the top n
    return [pt for dist, pt in points_with_distances[:n]]

def get_max_secant_dist_point(pareto_front: List[Tuple]) -> Tuple:
    """ 
    input: List[Tuple] - a list of (time, score, config, metrics) 
    output: (time, score, config), returning the "elbow" point using the normalized Kneedle method.
    """ 
    if len(pareto_front) < 3:
        return pareto_front[0] if pareto_front else None

    sorted_front = sorted(pareto_front, key=lambda x: x[0])

    min_time = sorted_front[0][0]
    max_time = sorted_front[-1][0]
    
    min_score = sorted_front[0][1]
    max_score = sorted_front[-1][1]

    if max_time == min_time or max_score == min_score:
        return sorted_front[0]

    max_secant_dist = -1
    best_pt = None

    for pt in sorted_front:
        time, score, config, metrics = pt
        
        norm_time = (time - min_time) / (max_time - min_time)
        norm_score = (score - min_score) / (max_score - min_score)
        
        dist = abs(norm_time - norm_score) / (2 ** 0.5)
        
        if dist > max_secant_dist:
            max_secant_dist = dist
            best_pt = pt

    return best_pt


def get_min_utopia_point(pareto_front: List[Tuple]) -> Tuple:
    """ 
    input: List[Tuple] - a list of (time, score, config, metrics) 
    output: (time, score, config), returning the model closest to the normalized Utopia point.
    """
    if not pareto_front:
        return None

    min_time = min(pt[0] for pt in pareto_front)
    max_time = max(pt[0] for pt in pareto_front)
    
    min_score = min(pt[1] for pt in pareto_front)
    max_score = max(pt[1] for pt in pareto_front)

    if max_time == min_time or max_score == min_score:
        return pareto_front[0]

    min_dist = float('inf')
    best_pt = None

    for pt in pareto_front:
        time, score = pt[0], pt[1]
        
        norm_time = (time - min_time) / (max_time - min_time)
        norm_score = (score - min_score) / (max_score - min_score)
        
        dist = ((norm_time - 0) ** 2 + (norm_score - 1) ** 2) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            best_pt = pt

    return best_pt



def find_param_in_dict(d: Dict[str, Any], param: str) -> Any:
    """Recursively search for a parameter in a dictionary."""
    if isinstance(d, dict):
        if param in d:
            return d[param]
        for v in d.values():
            found = find_param_in_dict(v, param)
            if found is not None:
                return found
    return None

def config_to_string(config: Dict) -> str:
    def get_val(section, param):
        if section not in config:
            return "?"
        
        found = find_param_in_dict(config[section], param)
        return str(found) if found is not None else "?"

    vec = get_val("data_model", "embedding_strategy")
    sim = get_val("data_model", "similarity_method")
    k = get_val("searcher", "k_neighbors")
    thresh = get_val("matcher", "similarity_threshold")
    return f"vec:{vec}-sim:{sim}-k:{k}-thresh:{thresh}"
