"""
Evaluation script
"""
import os
import json
from typing import Dict
from pathlib import Path

from nexus.utils.algo import Algo
from nexus.components import DATA_MODELS_REG, SEARCHERS_REG, MATCHERS_REG


def evaluate_on_dataset(
    best_config: Dict,
    test_file: str,
    output_path: str,
    proxy_score: bool = False,
    evaluate_gt: bool = True
) -> Dict:
    """
    Evaluate a configuration on the test set.
    
    Args:
        best_config: Best configuration from HPO
        test_file: Path to test dataset
        output_path: Directory to save evaluation results
        
    Returns:
        Dictionary with test metrics
    """
    
    output_path = Path(output_path)
    
    try:
        # Extract component names and params from config
        dm_name = best_config.get("data_model_name", "me_data")
        s_name = best_config.get("searcher_name", "tree")
        m_name = best_config.get("matcher_name", "greedy")
        
        dm_config = best_config.get("data_model", {})
        s_config = best_config.get("searcher", {})
        m_config = best_config.get("matcher", {})
        
        # Get params (handle both nested and flat structures)
        if "params" in dm_config:
            dm_params = dm_config["params"]
        else:
            dm_params = {k: v for k, v in dm_config.items() if k not in ["name", "path"]}
        
        if "params" in s_config:
            s_params = s_config["params"]
        else:
            s_params = {k: v for k, v in s_config.items() if k != "name"}
            
        if "params" in m_config:
            m_params = m_config["params"]
        else:
            m_params = {k: v for k, v in m_config.items() if k != "name"}
        
        # Setup data model with test file
        dm_params_with_path = dm_params.copy()
        dm_params_with_path['data'] = os.path.abspath(test_file)
        
        # Instantiate components
        data_model = DATA_MODELS_REG[dm_name](**dm_params_with_path)
        searcher = SEARCHERS_REG[s_name](**s_params)
        matcher = MATCHERS_REG[m_name](**m_params)
        
        # Run algorithm
        result = Algo(
            data_model=data_model,
            searcher=searcher,
            matcher=matcher
        ).exec(job_id="test_eval")
        
        # Compute metrics
        if evaluate_gt:
            acc, prec, rec, f1 = data_model.compute_metrics(result.matches)
        else:
            acc, prec, rec, f1 = None, None, None, None
        
        test_metrics = {
            "success": True,
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "acc": acc,
            "total_time": result.timing["total"],
            "num_matches": len(result.matches),
        }

        # Optionally compute proxy score if requested
        if proxy_score:
            proxy = data_model.compute_proxy(result.matches, reduction="sum")
            proxy_mean = data_model.compute_proxy(result.matches, reduction="mean")
            test_metrics["proxy_score"] = proxy
            test_metrics["proxy_score_mean"] = proxy_mean

        # Save test evaluation results
        test_results_path = output_path / "test_evaluation.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        return test_metrics
        
    except Exception as e:
        print(f"Error evaluating on test set: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "f1": None,
            "prec": None,
            "rec": None,
            "acc": None,
            "total_time": None,
            "proxy_score": None,
            "num_matches": 0
        }

