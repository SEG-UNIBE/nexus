
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os

from nexus.result_analysis.common import pretty_dataset_name

out_path = "output/exp_1"
file = "run_statistics.csv"
score_column = "f1"
score_column = "score"


# Load and concatenate two CSVs
csv_path = os.path.join(out_path, "run_statistics.csv")
# csv_path2 = os.path.join(out_path, "run_statistics_run1.csv")
df = pd.read_csv(csv_path)
# df2 = pd.read_csv(csv_path2)
# df = pd.concat([df1, df2], ignore_index=True)

# Get unique datasets
datasets = df['dataset'].unique()
n_datasets = len(datasets)

# Marker styles for each dataset
marker_styles = ['x', '.', '+', '*', 'v', '<', '>']

# Color gradient for HPO: green to blue
fig = plt.figure(figsize=(8, 6))
hpo_cmap = plt.get_cmap('winter')  # green to blue
colors = {'HPO': 'blue', 'Grid': 'red'}

handles = []
dataset_handles = []

normalize_time = True
normalize_score = True


for i, dataset in enumerate(datasets):
    subset = df[df['dataset'] == dataset].copy()
    # Normalize score to [0,1]
    if normalize_score:
        min_score = subset[score_column].min()
        max_score = subset[score_column].max()
        if max_score > min_score:
            subset['score_norm'] = (subset[score_column] - min_score) / (max_score - min_score)
        else:
            subset['score_norm'] = 0.0
    else:
        subset['score_norm'] = subset[score_column]


    # Normalize time to [0,1] per dataset and avoid zeros for log scale
    if normalize_time:
        min_time = subset['total_time'].min()
        max_time = subset['total_time'].max()
        if max_time > min_time:
            subset['total_time_norm'] = (subset['total_time'] - min_time) / (max_time - min_time)
        else:
            subset['total_time_norm'] = 0.0
    else:
        subset['total_time_norm'] = subset['total_time']

    subset['score_norm'] = subset['score_norm'].clip(lower=1e-6)
    subset['total_time_norm'] = subset['total_time_norm'].clip(lower=1e-6)

    for method in ['HPO', 'Grid']:
        method_subset = subset[subset['method'] == method]
        if method == 'HPO':
            # Color gradient for HPO based on alpha value
            if 'alpha' in method_subset.columns and method_subset['alpha'].notnull().any():
                alpha_vals = method_subset['alpha'].values
                # Normalize alpha to [0,1]
                alpha_norm = (alpha_vals - np.nanmin(alpha_vals)) / (np.nanmax(alpha_vals) - np.nanmin(alpha_vals) + 1e-8)
                hpo_colors = hpo_cmap(alpha_norm)
                h = plt.scatter(method_subset['total_time_norm'], method_subset['score_norm'],
                               c=hpo_colors, marker=marker_styles[i % len(marker_styles)],
                               label=f"{method} ({pretty_dataset_name(dataset)})", alpha=0.5, s=20)
            else:
                h = plt.scatter(method_subset['total_time_norm'], method_subset['score_norm'],
                               color=colors['HPO'], marker=marker_styles[i % len(marker_styles)],
                               label=f"{method} ({pretty_dataset_name(dataset)})", alpha=0.5, s=20)
        else:
            h = plt.scatter(method_subset['total_time_norm'], method_subset['score_norm'],
                           color=colors[method], marker=marker_styles[i % len(marker_styles)],
                           label=f"{method} ({pretty_dataset_name(dataset)})", alpha=0.5, s=20)
        # Only add one handle per method for legend
        if i == 0:
            handles.append(h)
    # Add dataset handle for legend
    dataset_handles.append(
        plt.Line2D([0], [0], marker=marker_styles[i % len(marker_styles)], color='gray',
                   markerfacecolor='gray', markeredgecolor='gray', markersize=8,
                   linestyle='None', label=pretty_dataset_name(dataset))
    )

plt.xlabel("Normalized Total Time (0-1)")
plt.ylabel("Normalized Score (0-1)")
plt.title("Normalized Score vs Normalized Total Time for All Datasets")


# Build legend
method_legend = plt.legend(handles, ['HPO', 'Grid'], title="Method", loc='lower right')
plt.gca().add_artist(method_legend)
# Show dataset legend in 2 rows (wrap) for readability
plt.legend(handles=dataset_handles, title="Dataset", loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)


# Add colorbar for HPO alpha gradient
hpo_alpha_vals = df[df['method'] == 'HPO']['alpha']
if not hpo_alpha_vals.empty:
    norm = mcolors.Normalize(vmin=hpo_alpha_vals.min(), vmax=hpo_alpha_vals.max())
    sm = plt.cm.ScalarMappable(cmap=hpo_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='vertical', pad=0.02, ax=plt.gca())
    cbar.set_label('HPO alpha value')

plt.tight_layout(rect=[0, 0.0, 1, 1])
plt.savefig(os.path.join(out_path, "score_vs_time_normalized.png"), dpi=300)
