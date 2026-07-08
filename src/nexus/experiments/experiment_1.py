"""
EXP 1: Dataset-Specific Tuning: optimize pareto front on each dataset

"""
import os
import pandas as pd
import numpy as np
import random
from nexus.core.optimise import NexusOptimizer
from nexus.utils import config_to_string
from nexus.utils.helper import get_pareto_front, get_max_secant_dist_point, get_min_utopia_point
import ray

RANDOM_SEED = 42
N_SAMPLES_OPTUNA = 60
WORKERS = min(64, os.cpu_count())
NR_REPETITIONS = 10

metric = ["proxy_score_sum", "total_time"]
mode = ["max", "min"]

ray.init(num_cpus=WORKERS)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    output_path = f"output/exp_1/"
    os.makedirs(output_path, exist_ok=True)

    datasets_map = {
        "hospitals": "src/assets/hospitals.csv", 
        "Apogames": "src/assets/Apogames.csv", 
        "warehouses": "src/assets/warehouses.csv",
        "random": "src/assets/random.csv", 
        "randomLoose": "src/assets/randomLoose.csv", 
        "randomTight": "src/assets/randomTight.csv", 
    }

    run_stats = []

    for dataset_name, dataset_path in datasets_map.items():
        dataset_output_path = os.path.join(output_path, f"{dataset_name}")
        os.makedirs(dataset_output_path, exist_ok=True)

        # Calculate n_models for grid parameters based on training set
        df = pd.read_csv(dataset_path, header=None)
        n_models = df.iloc[:, 0].nunique()

        # Part 1: Grid Configs 
        print(f"Running Grid Search...")
        
        grid_config = {
            "data_model": {
                "me_data": {
                    "path": [dataset_path],
                    "embedding_strategy": ["ld", "hd"],
                    "similarity_method": ["jaccard", "weighted"]
                }
            },
            "searcher": {
                "tree": {
                    "tree_method": ["kdtree"],
                    "k_neighbors": [int(n_models / 2), n_models, 2 * n_models]
                }
            },
            "matcher": {
                "greedy": {
                    "similarity_threshold": [0.25, 0.5, 0.75, 1.0]
                }
            }
        }

        grid_output_path = os.path.join(dataset_output_path, "grid")
        grid_tuner = NexusOptimizer(
            config=grid_config,
            output_path=grid_output_path,
            score_reduction_fct=lambda x: x["proxy"]
        )
        
        grid_results = grid_tuner.tune(
            n_samples=1,  # Grid search iterates all combinations
            algorithm="grid",
            num_workers=WORKERS,
            metric = "proxy", 
            mode = "max"
        )

        for result in grid_results:
            curr_config = result.config
            proxy = result.metrics.get(metric[0])
            total_time = result.metrics.get(metric[1])
            num_matches = result.metrics.get("num_matches")

            run_stats.append({
                "method": "Grid",
                "config": config_to_string(curr_config),
                "repetition": 0,
                "dataset": dataset_name,
                "score": proxy, 
                "total_time": total_time, 
                "num_matches": num_matches
            })

            # TODO: add repetitions

        for rep_i in range(NR_REPETITIONS):
            # Part 2: HPO (Optuna)
            print(f"Running Optuna HPO...")
            hpo_output_path = os.path.join(dataset_output_path, "optuna")
            
            hpo_config = {
                "data_model": {
                    "me_data": {
                        "path": [dataset_path],
                        "embedding_strategy": ["ld", "hd"],
                        "similarity_method": ["jaccard", "weighted"]
                    }
                },
                "searcher": {
                    "tree": {
                        "tree_method": ["kdtree"],
                        "k_neighbors": [1, 256]  # Range for Optuna
                    }
                },
                "matcher": {
                    "greedy": {
                        "similarity_threshold": [0.0, 1.0]  # Range for Optuna
                    }
                }
            }

            tuner = NexusOptimizer(
                config=hpo_config,
                output_path=hpo_output_path,
                score_reduction_fct=None
            )

            results = tuner.tune(
                n_samples=N_SAMPLES_OPTUNA,
                algorithm="optuna",
                num_workers=WORKERS,
                metric = metric, 
                mode = mode,
                seed=RANDOM_SEED+rep_i
            )

            pareto_front = get_pareto_front(results, s_metric=metric[0], t_metric=metric[1])
            for (time, score, pareto_config, pareto_metrics) in pareto_front:                
                run_stats.append({
                    "method": "HPO",
                    "config": config_to_string(pareto_config),
                    "repetition": rep_i,
                    "dataset": dataset_name,
                    "score": score,
                    "total_time": time,
                    "num_matches": pareto_metrics.get("num_matches", 0)
                })

            max_sec_dist_point = get_max_secant_dist_point(pareto_front)
            min_utopia_point = get_min_utopia_point(pareto_front) 

            if max_sec_dist_point is not None:
                metrics = max_sec_dist_point[3]
                run_stats.append({
                    "method": "HPO_max_sec_dist",
                    "config": config_to_string(max_sec_dist_point[2]),
                    "repetition": rep_i,
                    "dataset": dataset_name,
                    "score": max_sec_dist_point[1],
                    "total_time": max_sec_dist_point[0],
                    "num_matches": metrics.get("num_matches", 0),
                })

            if min_utopia_point is not None:
                metrics = min_utopia_point[3]
                run_stats.append({
                    "method": "HPO_min_utopia_dist",
                    "config": config_to_string(min_utopia_point[2]),
                    "repetition": rep_i,
                    "dataset": dataset_name,
                    "score": min_utopia_point[1],
                    "total_time": min_utopia_point[0],
                    "num_matches": metrics.get("num_matches", 0),
                })

            # Save run statistics
            run_stats_df = pd.DataFrame(run_stats)
            run_stats_df.to_csv(os.path.join(output_path, "run_statistics.csv"), index=False)
            
            cols = [
                "score",
                "total_time",
                "num_matches",
            ]

            grouped = run_stats_df.groupby(["method", "dataset", "repetition"])[cols].agg(['max', 'std']).reset_index()
            grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grouped.columns]
            grouped = grouped.sort_values("dataset")
            grouped.to_csv(os.path.join(output_path, "summary.csv"), index=False)

    latex_table = run_stats_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
    latex_path = os.path.join(output_path, "results_table.txt")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")

if __name__ == "__main__":
    main()