# Nexus - N-Way Model Matching Framework

This repository contains the code for the paper `Taming the Trade-off: Efficiency vs. Quality in N-Way Model Matching`. 

## Install

1. Clone the repository:
	```
	git clone https://github.com/user-34234/nexus.git
	cd nexus
	```
2. (Recommended) Create and activate a virtual environment:
	```
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. Install dependencies (using [uv](https://github.com/astral-sh/uv)):
	```
	uv sync
	```
	This will install all dependencies specified in your `pyproject.toml` (and `uv.lock` if present).

## Architecture

The main project structure is as follows:

- `src/nexus/`
  - `experiments/` — Scripts to run experiments (e.g., `experiment_1.py`, `experiment_2.py`)
  - `analysis/` — Analysis scripts for results and datasets (e.g., `analyse_exp1.py`, `analyse_dataset.py`)
  - `result_analysis/` — Scripts to generate plots and tables from experiment results
  - `core/`, `components/`, `utils/` — Core logic, reusable components, and utility functions
- `src/assets/` — Input datasets
- `output/` — Generated results, tables, and plots
- `pyproject.toml` — Project metadata


## Experiments

The following parts specify how to run the experiments 1 and 2 from the paper 

### Datasets

All datasets are stored under src/assets/*.csv

### Experiment 1 - Dataset-Specific Tuning

To run experiment 1 run the following lines

```
PYTHONPATH="$(pwd)/src" python3 src/nexus/experiments/experiment_1.py
```

Once this is finished, run 

```
PYTHONPATH="$(pwd)/src" python3 src/nexus/result_analysis/analyse_exp1.py
```

to generate the plots and tables

### Experiment 2 - Cross-Dataset Generalization


To run the second experiment run

```
PYTHONPATH="$(pwd)/src" python3 src/nexus/experiments/experiment_2.py
```

Once finished, the plots and tables can be generated with 

```
PYTHONPATH="$(pwd)/src" python3 src/nexus/result_analysis/analyse_exp2.py
```

