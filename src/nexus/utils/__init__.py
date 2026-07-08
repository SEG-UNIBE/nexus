"""
Nexus utility modules.

NOTE: nexus.utils.evaluate is NOT imported eagerly here to avoid a circular
import chain:
  nexus.components → data_model → nexus.utils → evaluate → nexus.components
Use ``from nexus.utils.evaluate import plot_algorithm_ranking_and_heatmap``
directly where needed.
"""
from nexus.utils.split_dataset import generate_test_eval_split
from nexus.utils.helper import config_to_string


__all__ = [
    'compute_overlap',
    'plot_algorithm_ranking_and_heatmap',
    'generate_test_eval_split',
    'config_to_string'
]
