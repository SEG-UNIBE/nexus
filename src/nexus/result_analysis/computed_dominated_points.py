import os
import pandas as pd
import numpy as np

from nexus.result_analysis.common import pretty_dataset_name


def is_dominated(grid_point, hpo_frontier):
    """
    Check if a grid point (time, score) is dominated by any point in the HPO frontier.
    A point is dominated if there exists a frontier point with lower/equal time AND higher/equal score.
    """
    grid_time, grid_score = grid_point
    for hpo_time, hpo_score in hpo_frontier:
        if hpo_time <= grid_time and hpo_score >= grid_score:
            # Check if at least one is strictly better
            if hpo_time < grid_time or hpo_score > grid_score:
                return True
    return False


DATASETS = [
    "warehouses",
    "hospitals",
    "Apogames",
    "random",
    "randomLoose",
    "randomTight",
]


def compute_pct_dominated(csv_path):
    df = pd.read_csv(csv_path)
    summaries = []

    for dataset in DATASETS:
        dataset_df = df[df["dataset"] == dataset]
        if dataset_df.empty:
            continue

        dominated_total = 0
        grid_total = 0

        for rep in sorted(dataset_df["repetition"].unique()):
            rep_df = dataset_df[dataset_df["repetition"] == rep]

            grid_df = rep_df[rep_df["method"] == "Grid"]
            hpo_df = rep_df[rep_df["method"] == "HPO"]

            grid_points = list(zip(grid_df["total_time"], grid_df["score"]))
            hpo_points = list(zip(hpo_df["total_time"], hpo_df["score"]))

            if not grid_points or not hpo_points:
                continue

            dominated_total += sum(1 for gp in grid_points if is_dominated(gp, hpo_points))
            grid_total += len(grid_points)

        if grid_total > 0:
            summaries.append({
                "Dataset": dataset,
                "Pct Dominated": (dominated_total / grid_total) * 100,
            })

    return pd.DataFrame(summaries)


def main():
    exp1_30_csv = "output/exp_1_30_iterations/run_statistics.csv"
    exp1_csv = "output/exp_1/run_statistics.csv"
    out_dir = "output/combined"
    out_csv = os.path.join(out_dir, "exp1_domination_comparison.csv")
    out_tex = os.path.join(out_dir, "exp1_domination_comparison.tex")

    exp1_30_df = compute_pct_dominated(exp1_30_csv).rename(columns={"Pct Dominated": "exp1_30"})
    exp1_df = compute_pct_dominated(exp1_csv).rename(columns={"Pct Dominated": "exp1"})

    results_df = pd.merge(exp1_30_df, exp1_df, on="Dataset", how="outer")
    results_df["Dataset"] = pd.Categorical(results_df["Dataset"], categories=DATASETS, ordered=True)
    results_df = results_df.sort_values("Dataset").reset_index(drop=True)
    results_df["Dataset"] = results_df["Dataset"].astype(str)

    results_out = results_df.copy()
    results_out["Dataset"] = results_out["Dataset"].apply(pretty_dataset_name)

    total_row = {
        "Dataset": "Total Mean",
        "exp1_30": results_df["exp1_30"].mean(),
        "exp1": results_df["exp1"].mean(),
    }

    results_out.to_csv(out_csv, index=False)
    print(f"Domination comparison CSV saved to {out_csv}")

    # Generate transposed LaTeX table: settings as rows, datasets as columns
    dataset_cols = [ds for ds in DATASETS if ds in set(results_df["Dataset"])]

    def format_pct(value):
        if pd.isna(value):
            return "-"
        return f"{value:.1f}\\%"

    col_spec = "l" + "r" * (len(dataset_cols) + 1)  # Setting + dataset columns + Total Mean
    header_cells = [
        r"{\bf Setting}",
    ] + [
        rf"{{\bf {pretty_dataset_name(ds)}}}" for ds in dataset_cols
    ] + [
        r"{\bf Total Mean}"
    ]

    table_lines = [
        "{",
        r"\setlength{\tabcolsep}{3pt}",
        r"\scriptsize",
        "",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    exp1_30_values = []
    exp1_values = []
    for ds in dataset_cols:
        ds_row = results_df[results_df["Dataset"] == ds]
        exp1_30_values.append(ds_row["exp1_30"].iloc[0] if not ds_row.empty else np.nan)
        exp1_values.append(ds_row["exp1"].iloc[0] if not ds_row.empty else np.nan)

    row_30 = [r"{\bf 30 iter.}"] + [format_pct(v) for v in exp1_30_values] + [format_pct(total_row["exp1_30"])]
    row_60 = [r"{\bf 60 iter.}"] + [format_pct(v) for v in exp1_values] + [format_pct(total_row["exp1"])]

    table_lines.append(" & ".join(row_30) + r" \\")
    table_lines.append(" & ".join(row_60) + r" \\")
    
    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "}",
    ])
    
    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write("\n".join(table_lines) + "\n")
    
    print(f"Domination comparison LaTeX table saved to {out_tex}")
    print("\nDomination results:")
    print(results_out.to_string(index=False))


if __name__ == "__main__":
    main()
