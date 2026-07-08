import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Ellipse
from matplotlib.ticker import MaxNLocator

from nexus.result_analysis.common import configure_plotting, pretty_dataset_name

configure_plotting()

# --- Plotting Options ---
PLOT_STD_CIRCLES = True 

# --- Configuration & Data Loading ---
out_path = "output/exp_2/"
file = "run_statistics.csv"
csv_path = os.path.join(out_path, file)
df = pd.read_csv(csv_path)
df['dataset'] = df['dataset'].str.replace('holdout_', '')

df_exp1 = pd.read_csv(os.path.join("output/exp_1/", "run_statistics.csv"))
df_exp1['dataset'] = df_exp1['dataset'].str.replace('holdout_', '')
df_grid = df_exp1[df_exp1['method'] == 'Grid']
df_hpo = df_exp1[df_exp1['method'] == 'HPO']
df = pd.concat([df, df_grid, df_hpo])


dataset_to_N_map = {
    'warehouses': 16, 
    'hospitals': 8, 
    'Apogames': 20, 
    'random': 100, 
    'randomLoose': 100, 
    'randomTight': 100,
}

dataset_order = ['warehouses', 'hospitals', 'Apogames', 'random', 'randomLoose', 'randomTight']
all_datasets = list(df['dataset'].unique())
datasets = [ds for ds in dataset_order if ds in all_datasets]
remaining = [ds for ds in all_datasets if ds not in dataset_order]
if remaining:
    datasets.extend(remaining)

grey_palette = plt.cm.Greys(np.linspace(0.25, 0.85, 8))

# --- Colors and Markers ---
sns_colors = sns.color_palette("Set2", n_colors=8)
grey_color = '0.8'
dark_grey_color = '0.6'
colors = {
    'HPO': grey_palette[1],
    'Grid': grey_palette[2],
    'HPO_min_utopia_dist_global': sns_colors[1],
    'HPO_utopia_exp1': sns_colors[2],
    'high-dim/jaccard:0.75/k:N': grey_palette[4],
    'high-dim/weighted/k:N': grey_palette[0],
    'Pareto Front': grey_palette[6],
} 
markers = {
    'Grid': 'o',
    'HPO': 'o',
    'HPO_min_utopia_dist_global': 'P',
    'HPO_utopia_exp1': '^',
    'high-dim/jaccard:0.75/k:N': 'D',
    'high-dim/weighted/k:N': 'D',
}

target_methods = [
    "HPO_min_utopia_dist_global"
]

# --- Data Aggregation ---

def most_frequent(series):
    if series.empty:
        return "N/A"
    return series.value_counts().idxmax()

df_exp1_utopia = df_exp1[df_exp1['method'] == "HPO_min_utopia_dist"].copy()
if not df_exp1_utopia.empty:
    df_exp1_utopia = df_exp1_utopia.groupby(['dataset', 'method']).agg(
        time_mean=('total_time', 'mean'),
        time_std=('total_time', 'std'),
        score_mean=('score', 'mean'),
        score_std=('score', 'std'),
        config_mode=('config', most_frequent)
    ).reset_index()

df_target_methods = df[df['method'].isin(target_methods)].copy()

df_target_methods = df_target_methods.groupby(['dataset', 'method']).agg(
    time_mean=('total_time', 'mean'),
    time_std=('total_time', 'std'),
    score_mean=('score', 'mean'),
    score_std=('score', 'std'),
    config_mode=('config', most_frequent)
).reset_index()

df_target_methods = df_target_methods.rename(columns={
    'total_time_mean': 'time_mean',
    'total_time_std': 'time_std',
})
baseline_methods = []

for dataset in df['dataset'].unique():
    grid_subset = df[(df['dataset'] == dataset) & (df['method'] == 'Grid')]
    # Get baseline grid configurations
    if not grid_subset.empty:
        N = dataset_to_N_map.get(dataset)
        if N:
            jaccard_prefix = f"vec:hd-sim:jaccard-k:{N}-thresh:0.75"
            weighted_prefix = f"vec:hd-sim:weighted-k:{N}-thresh:"

            jaccard_matches = grid_subset[grid_subset['config'].astype(str).str.startswith(jaccard_prefix)]
            if not jaccard_matches.empty:
                baseline_methods.append({
                    'dataset': dataset,
                    'method': 'high-dim/jaccard:0.75/k:N',
                    'time_mean': jaccard_matches['total_time'].mean(),
                    'score_mean': jaccard_matches['score'].mean(),
                    'time_std': jaccard_matches['total_time'].std(),
                    'score_std': jaccard_matches['score'].std(),
                    'config_mode': jaccard_matches['config'].iloc[0]
                })

            weighted_matches = grid_subset[grid_subset['config'].astype(str).str.startswith(weighted_prefix)]
            if not weighted_matches.empty:
                baseline_methods.append({
                    'dataset': dataset,
                    'method': 'high-dim/weighted/k:N',
                    'time_mean': weighted_matches['total_time'].mean(),
                    'score_mean': weighted_matches['score'].mean(),
                    'time_std': weighted_matches['total_time'].std(),
                    'score_std': weighted_matches['score'].std(),
                    'config_mode': weighted_matches['config'].iloc[0]
                })

baseline_methods = pd.DataFrame(baseline_methods)
all_special_dfs = pd.concat([df_target_methods, baseline_methods]).reset_index(drop=True)


# --- Aggregated Plotting ---
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    
    # 1. Plot Grid points
    grid_subset = df[(df['dataset'] == dataset) & (df['method'] == 'Grid')]
    if not grid_subset.empty:
        ax.scatter(grid_subset['total_time'], grid_subset["score"],
                   color=colors['Grid'], marker=markers['Grid'],
                   label="Grid", alpha=0.2, s=35, edgecolor='w', linewidth=0.5, zorder=3)

    # 2. Plot all HPO points and Pareto Band
    hpo_subset_all_reps = df[(df['dataset'] == dataset) & (df['method'] == 'HPO')].copy()
    if not hpo_subset_all_reps.empty:
        ax.scatter(hpo_subset_all_reps['total_time'], hpo_subset_all_reps["score"],
                   color=colors['HPO'], marker=markers['HPO'],
                   label="HPO Sample", alpha=0.2, s=35, edgecolor='w', linewidth=0.5, zorder=2)

        all_pareto_fronts = []
        for rep in hpo_subset_all_reps['repetition'].unique():
            rep_subset = hpo_subset_all_reps[hpo_subset_all_reps['repetition'] == rep]
            if rep_subset.empty: continue
            
            points_sorted = rep_subset.sort_values(by=['total_time', 'score'], ascending=[True, False])
            pareto_front_points = []
            max_score_so_far = -float('inf')
            for _, row in points_sorted.iterrows():
                if row['score'] > max_score_so_far:
                    pareto_front_points.append((row['total_time'], row['score']))
                    max_score_so_far = row['score']
            
            if pareto_front_points:
                all_pareto_fronts.append(np.array(pareto_front_points))

        if all_pareto_fronts:
            min_time = min(front[0, 0] for front in all_pareto_fronts if len(front) > 0)
            max_time = max(front[-1, 0] for front in all_pareto_fronts if len(front) > 0)
            
            if min_time < max_time:
                common_time_axis = np.linspace(min_time, max_time, 200)
                interpolated_scores = [np.interp(common_time_axis, front[:, 0], front[:, 1]) for front in all_pareto_fronts if len(front) > 0]
                
                if interpolated_scores:
                    scores_array = np.array(interpolated_scores)
                    mean_scores = np.mean(scores_array, axis=0)
                    lower_bound = np.percentile(scores_array, 0, axis=0)
                    upper_bound = np.percentile(scores_array, 100, axis=0)
                    
                    ax.plot(common_time_axis, mean_scores, color=colors['Pareto Front'], lw=2, label='Mean Pareto Front', zorder=4, alpha=0.2)
                    ax.fill_between(common_time_axis, lower_bound, upper_bound, color=colors['Pareto Front'], alpha=0.2, label='Pareto Frontier Band', zorder=1)

    # 3. Plot Average Utopia and Secant points from aggregated data
    if not all_special_dfs.empty:
        dataset_special_df = all_special_dfs[all_special_dfs['dataset'] == dataset]
        
        utopia_avg_row = dataset_special_df[dataset_special_df['method'] == 'HPO_min_utopia_dist_global']
        if not utopia_avg_row.empty:
            time_mean = utopia_avg_row['time_mean'].values[0]
            score_mean = utopia_avg_row['score_mean'].values[0]
            ax.scatter(time_mean, score_mean,
                       color=colors['HPO_min_utopia_dist_global'], marker=markers['HPO_min_utopia_dist_global'],
                       label="Avg. Utopia", s=180, edgecolor='black', linewidth=1.5, zorder=6)
            
            if PLOT_STD_CIRCLES and 'time_std' in utopia_avg_row.columns and 'score_std' in utopia_avg_row.columns:
                time_std = utopia_avg_row['time_std'].fillna(0).values[0]
                score_std = utopia_avg_row['score_std'].fillna(0).values[0]
                ellipse = Ellipse(xy=(time_mean, score_mean),
                                  width=time_std * 2, height=score_std * 2,
                                  facecolor=colors['HPO_min_utopia_dist_global'], 
                                  alpha=0.5, zorder=5, edgecolor=colors['HPO_min_utopia_dist_global'])
                ax.add_patch(ellipse)

        if not df_exp1_utopia.empty:
            dataset_exp1_utopia_df = df_exp1_utopia[df_exp1_utopia['dataset'] == dataset]
            if not dataset_exp1_utopia_df.empty:
                time_mean = dataset_exp1_utopia_df['time_mean'].values[0]
                score_mean = dataset_exp1_utopia_df['score_mean'].values[0]
                ax.scatter(time_mean, score_mean,
                           color=colors['HPO_utopia_exp1'], marker=markers['HPO_utopia_exp1'],
                           label="HPO Utopia Point (Exp 1)", s=180, edgecolor='black', linewidth=1.5, zorder=7)

        secant_avg_row = dataset_special_df[dataset_special_df['method'] == 'HPO_max_sec_dist_global']
        if not secant_avg_row.empty:
            time_mean = secant_avg_row['time_mean'].values[0]
            score_mean = secant_avg_row['score_mean'].values[0]
            ax.scatter(time_mean, score_mean,
                       color=colors['HPO_max_sec_dist_global'], marker=markers['HPO_max_sec_dist_global'],
                       label="Avg. Secant", s=180, edgecolor='black', linewidth=1.5, zorder=6)

            if PLOT_STD_CIRCLES and 'time_std' in secant_avg_row.columns and 'score_std' in secant_avg_row.columns:
                time_std = secant_avg_row['time_std'].fillna(0).values[0]
                score_std = secant_avg_row['score_std'].fillna(0).values[0]
                ellipse = Ellipse(xy=(time_mean, score_mean),
                                  width=time_std * 2, height=score_std * 2,
                                  facecolor=colors['HPO_max_sec_dist_global'], 
                                  alpha=0.5, zorder=5, edgecolor=colors['HPO_max_sec_dist_global'])
                ax.add_patch(ellipse)

    # 4. Highlight baseline configurations
    if dataset in dataset_to_N_map:
        N = dataset_to_N_map[dataset]
        jaccard_prefix = f"vec:hd-sim:jaccard-k:{N}-thresh:0.75"
        weighted_prefix = f"vec:hd-sim:weighted-k:{N}-thresh:"

        jaccard_match = grid_subset[grid_subset['config'].astype(str).str.startswith(jaccard_prefix)]
        if not jaccard_match.empty:
            ax.scatter(jaccard_match['total_time'], jaccard_match['score'],
                       color=colors['high-dim/jaccard:0.75/k:N'], marker=markers['high-dim/jaccard:0.75/k:N'], s=150, edgecolor='black', linewidth=1, zorder=5, alpha=0.2)

        weighted_match = grid_subset[grid_subset['config'].astype(str).str.startswith(weighted_prefix)]
        if not weighted_match.empty:
            ax.scatter(weighted_match['total_time'], weighted_match['score'],
                       color=colors["high-dim/weighted/k:N"], marker=markers['high-dim/weighted/k:N'], s=150, edgecolor='black', linewidth=1, zorder=5, alpha=0.2)

    # Subplot formatting
    ax.set_title(f"Holdout Dataset: {pretty_dataset_name(dataset)}")
    if i % 3 == 0: ax.set_ylabel("Weight Score")
    if i // 3 == 1: ax.set_xlabel("Total Time (s)")
    ax.tick_params(axis='both', which='both', length=5, width=1, colors='black')

# Final figure adjustments
for j in range(len(datasets), len(axes)):
    fig.delaxes(axes[j])

# fig.suptitle("Aggregated HPO Performance vs. Grid Search", y=0.96, weight='bold')
fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])

# --- Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Grid Samples', markerfacecolor=colors['Grid'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='HPO Samples on the Pareto Front', markerfacecolor=colors['HPO'], markersize=10, alpha=0.7),
    Polygon([[0,0]], facecolor=colors['Pareto Front'], alpha=0.2, label='Pareto Frontier Band'),
    Line2D([0], [0], color=colors['Pareto Front'], lw=2, label='Mean Pareto Front'),
    Line2D([0], [0], marker='D', color='w', label='Jaccard Baseline', markerfacecolor=colors['high-dim/jaccard:0.75/k:N'], markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='D', color='w', label='Weight Baseline', markerfacecolor=colors['high-dim/weighted/k:N'], markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker=markers["HPO_min_utopia_dist_global"], color='w', label=r'$\mathrm{HPO}_{utopia}$ (Exp 2)', markerfacecolor=colors['HPO_min_utopia_dist_global'], markeredgecolor='black', markersize=12),
    Line2D([0], [0], marker=markers['HPO_utopia_exp1'], color='w', label=r'$\mathrm{HPO}_{utopia}$ (Exp 1)', markerfacecolor=colors['HPO_utopia_exp1'], markeredgecolor='black', markersize=12),
]


legend_ax = fig.add_axes([0.05, 0.01, 0.90, 0.08])
legend_ax.axis('off')
legend_ax.legend(handles=legend_elements, loc='center', ncol=4, frameon=False)

# --- Save Figure ---
fig.savefig(os.path.join(out_path, "exp2_aggregated_plot.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_path, "exp2_aggregated_plot.svg"), bbox_inches='tight')

# --- Save Data Tables ---
if not all_special_dfs.empty:
    baselines_from_grid = all_special_dfs[~all_special_dfs['method'].isin(target_methods) & (all_special_dfs['method'] != 'Grid') & (all_special_dfs['method'] != 'HPO')].copy()

    weighted_baseline = baselines_from_grid[baselines_from_grid['method'].str.contains('weighted')].set_index('dataset')

    # Keep all HPO runs (not aggregated)
    hpo_all = df[df['method'].isin(target_methods)].copy()

    # Map baseline values to each run
    hpo_all['weighted_score_base'] = hpo_all['dataset'].map(weighted_baseline['score_mean'])

    # Compute score drop (%) per run (negative = worse, positive = better)
    hpo_all['score_drop_pct'] = (hpo_all['score'] / hpo_all['weighted_score_base'] - 1.0) * 100

    # Aggregate run-level statistics
    hpo_methods_df = hpo_all.groupby(['dataset', 'method']).agg(
        time_mean=('total_time', 'mean'),
        time_std=('total_time', 'std'),
        score_mean=('score', 'mean'),
        score_std=('score', 'std'),
        score_drop_weighted_mean=('score_drop_pct', 'mean'),
        score_drop_weighted_std=('score_drop_pct', 'std'),
        config=('config', most_frequent)
    ).reset_index()

    # Time speedups as ratio of means: baseline_time / mean(run_time)
    weighted_time_base_agg = hpo_methods_df['dataset'].map(weighted_baseline['time_mean'])

    hpo_methods_df['time_speedup_weighted_mean'] = np.where(
        hpo_methods_df['time_mean'] != 0,
        weighted_time_base_agg / hpo_methods_df['time_mean'],
        np.nan
    )
    # Keep std unset for ratio-of-means speedups unless a dedicated estimator is used.
    hpo_methods_df['time_speedup_weighted_std'] = np.nan

    # Calculate Total Means
    new_rows = []
    for method in target_methods:
        method_df = hpo_methods_df[hpo_methods_df['method'] == method]
        if not method_df.empty:
            mean_row = {
                'dataset': 'Total Mean',
                'method': method,
                'time_mean': method_df['time_mean'].mean(),
                'time_std': method_df['time_std'].mean(),
                'score_mean': method_df['score_mean'].mean(),
                'score_std': method_df['score_std'].mean(),
                'time_speedup_weighted_mean': method_df['time_speedup_weighted_mean'].mean(),
                'time_speedup_weighted_std': method_df['time_speedup_weighted_std'].mean(),
                'score_drop_weighted_mean': method_df['score_drop_weighted_mean'].mean(),
                'score_drop_weighted_std': method_df['score_drop_weighted_std'].mean(),
            }
            new_rows.append(pd.DataFrame([mean_row]))

    if new_rows:
        final_df = pd.concat([hpo_methods_df] + new_rows, ignore_index=True)
    else:
        final_df = hpo_methods_df.copy()

    if not final_df.empty:
        config_df = final_df[['dataset', 'method', 'config']].copy()
        config_df.loc[config_df['dataset'] == 'Total Mean', 'config'] = ''
        config_df = config_df.rename(columns={'dataset': 'Dataset', 'method': 'Method', 'config': 'Configuration'})

        config_df['Method'] = config_df['Method'].replace({
            'HPO_min_utopia_dist_global': r'$HPO_{utopia}',
        })

        final_df['method'] = final_df['method'].replace({
            'HPO_min_utopia_dist_global': r'$HPO_{utopia}',
        })

        def format_mean_std_compact(mean_val, std_val):
            if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0.005:
                return f"{mean_val:.2f}({std_val:.2f})"
            if pd.notna(mean_val):
                return f"{mean_val:.2f}"
            return ""

        final_df['time'] = final_df.apply(
            lambda row: format_mean_std_compact(row.get('time_mean'), row.get('time_std')),
            axis=1
        )
        final_df['score'] = final_df.apply(
            lambda row: format_mean_std_compact(row.get('score_mean'), row.get('score_std')),
            axis=1
        )
        final_df['Speedup'] = final_df['time_speedup_weighted_mean'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
        final_df['Score Delta (%)'] = final_df.apply(
            lambda row: format_mean_std_compact(row.get('score_drop_weighted_mean'), row.get('score_drop_weighted_std')),
            axis=1
        )

        final_df = final_df.rename(columns={
            'method': 'Method',
            'time': 'Time (s)',
            'score': 'Score',
            'dataset': 'Dataset',
        })


        latex_cols = ['Dataset', 'Time (s)', 'Score', 'Speedup', 'Score Delta (%)']
        present_latex_cols = [c for c in latex_cols if c in final_df.columns]
        final_latex_df = final_df[present_latex_cols].copy()

        sort_order = {name: idx for idx, name in enumerate(dataset_order)}
        sort_order['Total Mean'] = len(dataset_order)
        final_latex_df['_sort_key'] = final_latex_df['Dataset'].map(sort_order).fillna(999)
        final_latex_df = final_latex_df.sort_values('_sort_key').drop(columns=['_sort_key'])

        output_csv_path = os.path.join(out_path, "exp2_aggregated_results.csv")
        output_tex_path = os.path.join(out_path, "exp2_aggregated_results.tex")
        
        final_latex_df.to_csv(output_csv_path, index=False)

        table_lines = [
            r'\begin{tabular}{l | S[table-format=2.2(2.2)] | S[table-format=2.2(1.2)] | r | S[table-format=-2.2(2.2)]}',
            r'\toprule',
            r'\textbf{Dataset} & \textbf{Time (s)} & \textbf{Score} & \multicolumn{2}{c}{\textbf{Weighted-Baseline}} \\',
            r' & & & \textbf{Speedup} & \textbf{Score $\Delta$ (\%)} \\',
            r'\midrule',
        ]

        total_row = None
        for _, row in final_latex_df.iterrows():
            dataset = row['Dataset']
            dataset_label = dataset if dataset == 'Total Mean' else pretty_dataset_name(dataset)
            line = f"{dataset_label} & {row['Time (s)']} & {row['Score']} & {row['Speedup']} & {row['Score Delta (%)']} \\\\"
            if dataset == 'Total Mean':
                total_row = line
            else:
                table_lines.append(line)

        if total_row is not None:
            table_lines.append(r'\midrule')
            table_lines.append(total_row)

        table_lines.extend([
            r'\bottomrule',
            r'\end{tabular}'
        ])

        with open(output_tex_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(table_lines) + '\n')
        print(f"\nTable saved to {output_csv_path} and {output_tex_path}")

        config_output_tex_path = os.path.join(out_path, "exp2_configurations.tex")
        config_df_for_tex = config_df.copy()
        config_df_for_tex['Dataset'] = config_df_for_tex['Dataset'].apply(
            lambda ds: ds if ds == 'Total Mean' else pretty_dataset_name(ds)
        )
        config_df_for_tex.set_index(['Dataset', 'Method']).to_latex(
            config_output_tex_path,
            index=True,
            escape=False,
            na_rep='-',
            multirow=True,
        )
        print(f"Configuration table saved to {config_output_tex_path}")

    print(f"Aggregated plot and data tables saved to {out_path}")
    
# --- Configuration Frequency Tables ---
df = df[df['method'].isin(target_methods)]

hpo_configs_df = df[df['method'].isin(target_methods)]
if not hpo_configs_df.empty:
    config_counts_all = hpo_configs_df.groupby('config').agg(
        Count=('config', 'size'),
    ).reset_index().rename(columns={'config': 'Configuration'})
    config_counts_all = config_counts_all.sort_values('Count', ascending=False)
    config_counts_all.to_latex(os.path.join(out_path, 'exp2_hpo_config_counts_all.tex'), index=False, na_rep='-', float_format="%.2f")

aggregated_hpo_configs = all_special_dfs[all_special_dfs['method'].isin(target_methods)]
if not aggregated_hpo_configs.empty:
    config_counts_agg = aggregated_hpo_configs.groupby('config_mode').agg(
        Count=('config_mode', 'size'),
    ).reset_index().rename(columns={'config_mode': 'Configuration'})
    config_counts_agg = config_counts_agg.sort_values('Count', ascending=False)
    config_counts_agg.to_latex(os.path.join(out_path, 'exp2_hpo_config_counts_aggregated.tex'), index=False, na_rep='-', float_format="%.2f")

import re
def parse_dim_and_k(config):
    if "ld-sim" in config:
        dim = "low-dim"
    elif "hd-sim" in config:
        dim = "high-dim"
    else:
        dim = "other"
    k_match = re.search(r'k:(\d+)', config)
    k = int(k_match.group(1)) if k_match else None
    return pd.Series([dim, k])

# Apply parsing to config_counts_all
config_counts_all[['dim', 'k']] = config_counts_all['Configuration'].apply(parse_dim_and_k)

unique_k = sorted(config_counts_all["k"].dropna().unique())

# Map dim to y positions for transposed plot
y_map = {"low-dim": -0.2, "high-dim": 0.2}
config_counts_all["y"] = config_counts_all["dim"].map(y_map)

# Use seaborn and improved aesthetics
import seaborn as sns
palette = sns.color_palette("Set2", n_colors=2)

plt.clf()
fig, ax = plt.subplots(figsize=(8, 2.5))

sns.scatterplot(
    data=config_counts_all,
    x="k",
    y="y",
    size="Count",
    sizes=(20, 1200),
    hue="dim",
    palette=palette,
    alpha=0.8,
    edgecolor="k",
    ax=ax,
    legend=False
)

# Custom axes: x is true numeric k (real spacing), with regular increasing ticks
if unique_k:
    k_min, k_max = min(unique_k), max(unique_k)
    k_pad = max(1, int((k_max - k_min) * 0.03))
    ax.set_xlim(k_min - k_pad, k_max + k_pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=16, integer=True, min_n_ticks=12))

ax.set_yticks([-0.2, 0.2])
ax.set_yticklabels(["low-dim", "high-dim"], rotation=90, va='center')
ax.set_ylim(-0.5, 0.5)

# Add subtle center indicators on the y-axis for each category band
for y_center in (-0.2, 0.2):
    ax.plot(
        [0.0, -0.01],
        [y_center, y_center],
        transform=ax.get_yaxis_transform(),
        color='black',
        lw=1.0,
        clip_on=False,
        zorder=6,
    )

ax.set_xlabel("k Value (Neighbours)")
ax.set_ylabel("Vectorization Method")
ax.grid(False)

# Legend note for numeric annotations
occurrence_note = Line2D(
    [], [],
    linestyle='-',
    color='grey',
    lw=1.0,
    label='Number of occurrences'
)
ax.legend(
    handles=[occurrence_note],
    loc='upper right',
    frameon=False,
    handlelength=1.2,
    handletextpad=0.4,
    labelcolor='grey',
    fontsize=12,
)

# Add count labels near circles
label_offset = 0.15
for _, row in config_counts_all.iterrows():
    ax.annotate(
        f"{int(row['Count'])}",
        xy=(row["k"], row["y"]),
        xytext=(row["k"], row["y"] + label_offset),
        textcoords='data',
        color='grey',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight="bold",
        rotation=0,
        arrowprops=dict(
            arrowstyle='-',
            color='grey',
            alpha=1.0,
            lw=1.0,
            shrinkA=0,
            shrinkB=6,
        ),
    )

plt.tight_layout()
plt.savefig(os.path.join(out_path, "exp2_config_hist.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_path, "exp2_config_hist.svg"), bbox_inches='tight')