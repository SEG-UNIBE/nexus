import os
import json
import hashlib
import numpy as np
from typing import Dict, Any, List

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler 
import optuna
import pandas as pd
import itertools

from optuna.samplers import NSGAIISampler, TPESampler, RandomSampler
from nexus.utils.algo import Algo
from nexus.components import DATA_MODELS_REG, SEARCHERS_REG, MATCHERS_REG


class OptunaSearchSpace:
    def __init__(self, experiment_config: Dict[str, Any]):
        self.experiment_config = experiment_config

    def _suggest_value(self, trial, prefix, name, values):
        if isinstance(values, list):
            is_numeric = len(values) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values)
            if is_numeric:
                min_val = min(values)
                max_val = max(values)
                if min_val == max_val:
                    return min_val
                if all(isinstance(x, int) for x in values):
                    return trial.suggest_int(f'{prefix}_{name}', int(min_val), int(max_val))
                else:
                    return trial.suggest_float(f'{prefix}_{name}', float(min_val), float(max_val))
            else:
                return trial.suggest_categorical(f'{prefix}_{name}', values)
        return values


    def __call__(self, trial) -> Dict[str, Any]:
        """Define search space dynamically based on trial suggestions."""
        config = {}
        
        # Data Model
        dm_config = self.experiment_config.get('data_model', {})
        dm_names = list(dm_config.keys())
        if dm_names:
            dm_name = trial.suggest_categorical('data_model_name', dm_names)
            dm_params = {}
            
            selected_dm_config = dm_config.get(dm_name, {})
            for param_name, param_values in selected_dm_config.items():
                if param_name == 'path':
                    continue 
                dm_params[param_name] = self._suggest_value(trial, 'dm', param_name, param_values)
            
            config['data_model'] = {'name': dm_name, 'params': dm_params}
        
        # Searcher
        searcher_config = self.experiment_config.get('searcher', {})
        searcher_names = list(searcher_config.keys())
        if searcher_names:
            s_name = trial.suggest_categorical('searcher_name', searcher_names)
            s_params = {}
            
            selected_s_config = searcher_config.get(s_name, {})
            for param_name, param_values in selected_s_config.items():
                s_params[param_name] = self._suggest_value(trial, 's', param_name, param_values)
            
            config['searcher'] = {'name': s_name, 'params': s_params}
        
        # Matcher (conditional parameter logic)
        matcher_config = self.experiment_config.get('matcher', {})
        matcher_names = list(matcher_config.keys())
        if matcher_names:
            m_name = trial.suggest_categorical('matcher_name', matcher_names)
            m_params = {}
            selected_m_config = matcher_config.get(m_name, {})
            similarity_method = config.get('data_model', {}).get('params', {}).get('similarity_method', None)
            for param_name, param_values in selected_m_config.items():
                if param_name == 'similarity_threshold':
                    if similarity_method == 'jaccard':
                        m_params[param_name] = self._suggest_value(trial, 'm', param_name, param_values)
                else:
                    m_params[param_name] = self._suggest_value(trial, 'm', param_name, param_values)
            config['matcher'] = {'name': m_name, 'params': m_params}
        return config


def objective_function(config: Dict[str, Any], dataset_paths: List[str], output_path: str, score_reduction_fct: callable = None):
    """Ray Tune objective function.
    
    Works with nested config structure:
    - data_model: {name: str, params: {...}}
    - searcher: {name: str, params: {...}}
    - matcher: {name: str, params: {...}}
    """
    all_dataset_metrics = []

    if "_bundle" in config:
        bundled = config.pop("_bundle")
        config.update(bundled)

    # Extract component configs from nested structure
    dm_config = config.get('data_model', {})
    s_config = config.get('searcher', {})
    m_config = config.get('matcher', {})
    
    dm_name = dm_config.get('name', 'me_data')
    s_name = s_config.get('name', 'tree')
    m_name = m_config.get('name', 'greedy')
    
    dm_params = dm_config.get('params', {})
    s_params = s_config.get('params', {})
    m_params = m_config.get('params', {})

    pipeline = f"{dm_name}+{s_name}+{m_name}"
    
    # Generate config_id hash (excluding dataset path)
    dm_params_for_hash = {k: v for k, v in dm_params.items() if k != 'path'}
    config_str = json.dumps({
        'data_model': dm_name,
        'data_model_params': dm_params_for_hash,
        'searcher': s_name,
        'searcher_params': s_params,
        'matcher': m_name,
        'matcher_params': m_params
    }, sort_keys=True)

    config_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    context = tune.get_context()
    trial_id = context.get_trial_id()

    config["datasets"] = dataset_paths

    config_output_path = os.path.join(output_path, f"{config_id}")
    os.makedirs(config_output_path, exist_ok=True)
    with open(os.path.join(config_output_path ,f"config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    trial_data = []

    for path in dataset_paths:
        abs_path = os.path.abspath(path)

        try:
            # instantiate components
            dm_params_with_path = dm_params.copy()
            dm_params_with_path['data'] = abs_path

            data_model = DATA_MODELS_REG[dm_name](**dm_params_with_path)
            searcher = SEARCHERS_REG[s_name](**s_params)
            matcher = MATCHERS_REG[m_name](**m_params)
    
            result = Algo(
                data_model=data_model,
                searcher=searcher, 
                matcher=matcher
            ).exec(job_id=f"{trial_id}")

            total_time = result.timing["total"]
            nr_items = len(data_model)

            try:
                acc, prec, rec, f1 = data_model.compute_metrics(result.matches)
            except Exception as e:
                acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0

            f1_val = 0.0 if (f1 is None or np.isnan(f1)) else float(f1)

            time_per_item = total_time / max(1, nr_items)
            logt = np.log1p(time_per_item)
            inv_time = 1.0 / (1.0 + logt)

            try:
                proxy_score = data_model.compute_proxy(result.matches, reduction="mean")
                proxy_score_sum = data_model.compute_proxy(result.matches, reduction="sum")
            except Exception as e:
                proxy_score = 0.0
                proxy_score_sum = 0.0

            dataset_metrics_dict = {
                "f1": f1_val,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "proxy": proxy_score,
                "proxy_score_sum": proxy_score_sum,
                "inv_time": inv_time,
                "time_per_element": time_per_item,
                "total_time": total_time,
                "num_matches": len(result.matches),
            }
            all_dataset_metrics.append(dataset_metrics_dict)

        except Exception as e:
            print(f"Error in trial for dataset {abs_path}: {e}")
            import traceback
            traceback.print_exc()
            all_dataset_metrics.append({
                "f1": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0,
                "proxy": 0.0, "proxy_score_sum": 0.0,
                "total_time": 0.0, "inv_time": 0.0, "time_per_element": 0.0, 
                "num_matches": 0
            })

    final_metrics = {}
    if all_dataset_metrics:
        metric_keys = all_dataset_metrics[0].keys()
        for key in metric_keys:
            final_metrics[key] = np.mean([d.get(key, 0) for d in all_dataset_metrics])

    if score_reduction_fct:
        final_metrics["reduced_score"] = score_reduction_fct(final_metrics)
    else:
        final_metrics["reduced_score"] = 0.0

    final_metrics["config_id"] = config_id
    final_metrics["pipeline"] = pipeline
    final_metrics["trial_id"] = trial_id
    final_metrics["trial_data"] = trial_data

    final_metrics["config"] = {
        "data_model_name": dm_name,
        "searcher_name": s_name,
        "matcher_name": m_name,
        "data_model": dm_config,
        "searcher": s_config,
        "matcher": m_config
    }

    tune.report(final_metrics)

class NexusOptimizer:
    def __init__(self, 
        config: str | dict, 
        output_path: str, 
        score_reduction_fct: callable = None
    ):
        if isinstance(config, dict):
            self.experiment_config = config
        else:
            with open(config, 'r') as f:
                self.experiment_config = json.load(f)

        self.score_reduction_fct = score_reduction_fct

        dm_config = self.experiment_config.get('data_model', {})
        for params in dm_config.values():
            if 'path' in params:
                dataset_paths = params['path']
    
        self.dataset_paths = [os.path.abspath(p) for p in dataset_paths]
        self.output_path = output_path


    def _build_grid_param_space(self) -> Dict[str, Any]:
        """Build a Ray Tune grid_search param_space with conditional deduplication."""
        
        def get_list(v):
            return v if isinstance(v, list) else [v]

        dm_cfg = self.experiment_config.get('data_model', {})
        s_cfg = self.experiment_config.get('searcher', {})
        m_cfg = self.experiment_config.get('matcher', {})

        # Handle multiple component names if present (fallback to first if not)
        dm_name = list(dm_cfg.keys())[0] if dm_cfg else "default_dm"
        s_name = list(s_cfg.keys())[0] if s_cfg else "default_s"
        m_name = list(m_cfg.keys())[0] if m_cfg else "default_m"

        dm_params_raw = dm_cfg.get(dm_name, {})
        s_params_raw = s_cfg.get(s_name, {})
        m_params_raw = m_cfg.get(m_name, {})

        keys = []
        val_lists = []

        # Extract parameters to build the Cartesian product
        for k, v in dm_params_raw.items():
            if k != 'path':
                keys.append(("data_model", k))
                val_lists.append(get_list(v))
                
        for k, v in s_params_raw.items():
            keys.append(("searcher", k))
            val_lists.append(get_list(v))
            
        for k, v in m_params_raw.items():
            keys.append(("matcher", k))
            val_lists.append(get_list(v))

        valid_configs = []
        seen_signatures = set()

        # Generate all combinations
        for combo in itertools.product(*val_lists):
            combo_dict = dict(zip(keys, combo))
            sim_method = combo_dict.get(("data_model", "similarity_method"))
            
            if sim_method != 'jaccard' and ("matcher", "similarity_threshold") in combo_dict:
                combo_dict[("matcher", "similarity_threshold")] = get_list(m_params_raw["similarity_threshold"])[0]

            sig = tuple(sorted(combo_dict.items()))
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            config = {
                "data_model": {"name": dm_name, "params": {}},
                "searcher": {"name": s_name, "params": {}},
                "matcher": {"name": m_name, "params": {}}
            }
            
            for (comp, param_name), param_val in combo_dict.items():
                config[comp]["params"][param_name] = param_val

            valid_configs.append(config)

        return {"_bundle": tune.grid_search(valid_configs)}
    def tune(self, 
            n_samples: int = 20, 
            algorithm: str = "optuna", # "random", "grid"
            num_workers: int = 1,
            param_space = None,
            metric = "score",
            mode = "max",
            sampler = "TPE", # "NSGAII"
            store_results = False,
            seed = 42
        ):
        is_multi_objective = isinstance(metric, list) and len(metric) > 1

        sampler = None
        search_alg = None
        optuna_space = OptunaSearchSpace(self.experiment_config)
        if is_multi_objective:
            print("Multi-objective detected: Disabling ASHAScheduler.")
            scheduler = None
        else:
            scheduler = ASHAScheduler(
                max_t=30,
                grace_period=5, 
                reduction_factor=2
            )
        sampler = None
        search_alg = None
        optuna_space = OptunaSearchSpace(self.experiment_config)

        if algorithm in ["optuna", "random"]:
            if algorithm == "random":
                sampler = optuna.samplers.RandomSampler(seed=seed)

            elif algorithm == "optuna":
                if sampler == "NSGAII":
                    sampler = NSGAIISampler(seed=seed)
                else:
                    sampler = TPESampler(
                        seed=seed,
                        multivariate=True
                    )

            search_alg = OptunaSearch(
                space=optuna_space, 
                metric=metric, 
                mode=mode, 
                sampler=sampler,
                seed=seed
            )
            
        elif algorithm == "grid":
            search_alg = None
            param_space = self._build_grid_param_space()

        
        if is_multi_objective and search_alg is not None:
            tc_metric = None
            tc_mode = None
        else:
            tc_metric = metric[0] if isinstance(metric, list) else metric
            tc_mode = mode[0] if isinstance(mode, list) else mode

        tuner = tune.Tuner(
            tune.with_parameters(
                objective_function, 
                dataset_paths=self.dataset_paths,
                output_path=os.path.abspath(self.output_path),
                score_reduction_fct=self.score_reduction_fct
            ),
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=tc_metric,
                mode=tc_mode,
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=n_samples,
                max_concurrent_trials=num_workers
            )
        )

        results = tuner.fit()

        # store df with score, f1, acc, prec, trial_id, config_id, dataset
        if store_results:
            rows = []
            for result in results._results:
                for trial_data in result.metrics["trial_data"]:
                    df_row = {
                        "score": trial_data.get("score", 0.0),
                        "f1": trial_data.get("f1", 0.0),
                        "acc": trial_data.get("acc", 0.0),
                        "prec": trial_data.get("prec", 0.0),
                        "rec": trial_data.get("rec", 0.0),
                        "dataset": trial_data.get("dataset", ""),
                        "logt": trial_data.get("logt", 0.0),
                        "time_per_element": trial_data.get("time_per_element", 0.0),
                        "inv_time": trial_data.get("inv_time", 0.0),
                        "total_time": trial_data.get("total_time", 0.0),
                        "load_time": trial_data.get("load_time", 0.0),
                        "embed_time": trial_data.get("embed_time", 0.0),
                        "search_time": trial_data.get("search_time", 0.0),
                        "match_time": trial_data.get("match_time", 0.0),
                        "trial_id": trial_data.get("trial_id", ""),
                        "config_id": trial_data.get("config_id", ""),
                        "num_matches": trial_data.get("num_matches", 0)
                    }
                    rows.append(df_row)
            
            df = pd.DataFrame(rows)
            print(df)
            df.to_csv(os.path.join(self.output_path, "results.csv"), index=False)

        return results