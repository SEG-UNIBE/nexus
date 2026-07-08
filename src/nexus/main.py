"""
Main entry point for nexus experiments.
"""
import os
from nexus.core.optimise import NexusOptimizer
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    """Run hyperparameter optimization base experiment"""
    config = {
        "data_model": {
            "me_data": {
                "path": [
                    "src/assets/Apogames.csv"
                ],
                "embedding_strategy": ["ld"],
                "similarity_method": ["jaccard"]
            }
        },
        "searcher": { 
            "tree": {
                "tree_method": ["kdtree"],
                "k_neighbors": [1,32]
            }
        },
        "matcher": {
            "greedy": {
                "similarity_threshold": [0.0, 1.0]
            }
        }
    }

    N_SAMPLES = 30
    WORKERS = max(1, (os.cpu_count() or 2) - 1)

    output_path = f"output/{timestamp}_main_test/"
    os.makedirs(output_path, exist_ok=True)

    tuner = NexusOptimizer(
        config=config,
        output_path=output_path,
        score_reduction_fct=lambda x: x["proxy"]
    )
    
    print(f"Starting Ray Tune experiment with {N_SAMPLES} samples on {len(tuner.dataset_paths)} datasets...")
    
    # Execute experiment
    results = tuner.tune(
        n_samples=N_SAMPLES, 
        algorithm="optuna", # ["optuna", "random", "grid"]
        num_workers=WORKERS,
        metric="proxy",
        mode="max"
    )

if __name__ == "__main__":
    main()
