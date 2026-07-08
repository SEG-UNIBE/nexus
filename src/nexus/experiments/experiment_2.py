"""
EXP 2: Cross-Dataset Generalization: Optimize on 5/6 datasets, evaluate on holdout dataset
"""
import os
import pandas as pd
import numpy as np
import random

from nexus.core.optimise import NexusOptimizer
from nexus.utils import config_to_string
from nexus.utils.eval_test import evaluate_on_dataset
from nexus.utils.helper import get_pareto_front, get_max_secant_dist_point, get_min_utopia_point

RANDOM_SEED = 42
N_SAMPLES_OPTUNA = 60
WORKERS = min(64, os.cpu_count())
NR_REPETITIONS = 10

metric = ["proxy_score_sum", "total_time"]
mode = ["max", "min"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    output_path = f"output/exp_2/"
    os.makedirs(output_path, exist_ok=True)

    datasets_map = {
        "holdout_hospitals": "src/assets/hospitals.csv", 
        "holdout_random": "src/assets/random.csv", 
        "holdout_randomLoose": "src/assets/randomLoose.csv", 
        "holdout_randomTight": "src/assets/randomTight.csv", 
        "holdout_Apogames": "src/assets/Apogames.csv", 
        "holdout_warehouses": "src/assets/warehouses.csv"
    }
    all_datasets = list(datasets_map.values())

    run_stats = []

    for rep_i in range(NR_REPETITIONS):
        for eval_name, eval_dataset in datasets_map.items():
            # all datasets except the holdout
            train_datasets = [k for k in all_datasets if k != eval_dataset]

            dataset_output_path = os.path.join(output_path, f"{eval_name}")
            os.makedirs(dataset_output_path, exist_ok=True)

            # Calculate n_models for grid parameters based on training set
            n_models_list = []
            for dataset_path in train_datasets:
                df = pd.read_csv(dataset_path, header=None)
                n_models = df.iloc[:, 0].nunique()
                
                n_models_list.append(n_models)
                # n_models_list.append(n_models//2)
                # n_models_list.append(n_models*2)

            # n_models_min = min(n_models_list)
            # n_models_max = max(n_models_list)
            # n_models = (n_models_min + n_models_max) // 2
            n_models_list = list(set(n_models_list))

            # Part 0: Grid search over eval dataset -> what gives best educated guess and average
            grid_config = {
                "data_model": {
                    "me_data": {
                        "path": train_datasets,
                        "embedding_strategy": ["ld", "hd"],
                        "similarity_method": ["jaccard", "weighted"]
                    }
                },
                "searcher": {
                    "tree": {
                        "tree_method": ["kdtree"],
                        "k_neighbors": n_models_list # [int(n_models / 2), n_models, 2 * n_models]
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

            grid_pareto_front = get_pareto_front(grid_results, s_metric=metric[0], t_metric=metric[1])

            grid_max_sec_dist_point = get_max_secant_dist_point(grid_pareto_front)
            grid_min_utopia_point = get_min_utopia_point(grid_pareto_front) 

            if grid_max_sec_dist_point is not None:
                grid_max_sec_dist_config_eval = evaluate_on_dataset(
                    best_config=grid_max_sec_dist_point[2],
                    test_file=eval_dataset,
                    output_path=grid_output_path,
                    proxy_score=True,
                    evaluate_gt=False
                )

                run_stats.append({
                    "method": "Grid_max_sec_dist",
                    "config": config_to_string(grid_max_sec_dist_point[2]),
                    "dataset": eval_name,
                    "repetition": rep_i,
                    "score": grid_max_sec_dist_config_eval.get("proxy_score", 0.0),
                    "total_time": grid_max_sec_dist_config_eval.get("total_time", 0.0),
                    "num_matches": grid_max_sec_dist_config_eval.get("num_matches", 0)
                })

            if grid_min_utopia_point is not None:
                grid_min_utopia_dist_config_eval = evaluate_on_dataset(
                    best_config=grid_min_utopia_point[2],
                    test_file=eval_dataset,
                    output_path=grid_output_path,
                    proxy_score=True,
                    evaluate_gt=False
                )

                run_stats.append({
                    "method": "Grid_min_utopia_dist",
                    "config": config_to_string(grid_min_utopia_point[2]),
                    "dataset": eval_name,
                    "repetition": rep_i,
                    "score": grid_min_utopia_dist_config_eval.get("proxy_score", 0.0),
                    "total_time": grid_min_utopia_dist_config_eval.get("total_time", 0.0),
                    "num_matches": grid_min_utopia_dist_config_eval.get("num_matches", 0)
                })

            print(f"Running HPO on global datasets")
            
            hpo_config = {
                "data_model": {
                    "me_data": {
                        "path": train_datasets,
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

            hpo_output_path = os.path.join(dataset_output_path, "hpo_global")

            tuner = NexusOptimizer(
                config=hpo_config,
                output_path=hpo_output_path,
                score_reduction_fct=None
            )

            hpo_results = tuner.tune(
                n_samples=N_SAMPLES_OPTUNA,
                algorithm="optuna",
                num_workers=WORKERS,
                metric = metric, 
                mode = mode,
                seed = RANDOM_SEED + rep_i
            )

            pareto_front = get_pareto_front(hpo_results, s_metric=metric[0], t_metric=metric[1])
            max_sec_dist_point = get_max_secant_dist_point(pareto_front)
            min_utopia_point = get_min_utopia_point(pareto_front) 

            if max_sec_dist_point is not None:
                max_sec_dist_config_eval = evaluate_on_dataset(
                    best_config=max_sec_dist_point[2],
                    test_file=eval_dataset,
                    output_path=hpo_output_path,
                    proxy_score=True,
                    evaluate_gt=False
                )
                
                run_stats.append({
                    "method": "HPO_max_sec_dist_global",
                    "config": config_to_string(max_sec_dist_point[2]),
                    "dataset": eval_name,
                    "repetition": rep_i,
                    "score": max_sec_dist_config_eval.get("proxy_score", 0.0),
                    "total_time": max_sec_dist_config_eval.get("total_time", 0.0),
                    "num_matches": max_sec_dist_config_eval.get("num_matches", 0)
                })

            if min_utopia_point is not None:
                min_utopia_dist_config_eval = evaluate_on_dataset(
                    best_config=min_utopia_point[2],
                    test_file=eval_dataset,
                    output_path=hpo_output_path,
                    proxy_score=True,
                    evaluate_gt=False
                )
                
                run_stats.append({
                    "method": "HPO_min_utopia_dist_global",
                    "config": config_to_string(min_utopia_point[2]),
                    "dataset": eval_name,
                    "repetition": rep_i,
                    "score": min_utopia_dist_config_eval.get("proxy_score", 0.0),
                    "total_time": min_utopia_dist_config_eval.get("total_time", 0.0),
                    "num_matches": min_utopia_dist_config_eval.get("num_matches", 0)
                })

            # Save run statistics
            run_stats_df = pd.DataFrame(run_stats)
            run_stats_df.to_csv(os.path.join(output_path, "run_statistics.csv"), index=False)
            
            # Aggregate HPO results
            cols = [
                "score",
                "total_time",
                "num_matches",
            ]

            grouped = run_stats_df.groupby(["method", "dataset"])[cols].agg(['mean', 'std']).reset_index()
            grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in grouped.columns]
            grouped = grouped.sort_values("dataset")
            grouped.to_csv(os.path.join(output_path, "summary.csv"), index=False)

    # Save LaTeX table
    latex_table = run_stats_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
    latex_path = os.path.join(output_path, "results_table.txt")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")

if __name__ == "__main__":
    main()