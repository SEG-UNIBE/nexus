import os
import re
import pandas as pd


EXP1_CSV = "output/exp_1/exp1_aggregated_results.csv"
EXP2_CSV = "output/exp_2/exp_aggregated_results.csv"
OUT_DIR = "output/combined"
OUT_TEX = "exp1_exp2_combined_table.tex"


DATASET_ORDER = [
    "warehouses",
    "hospitals",
    "Apogames",
    "random",
    "randomLoose",
    "randomTight",
]

DATASET_LABELS = {
    "warehouses": "Warehouse~",
    "hospitals": "Hospital~",
    "Apogames": "Apo-Games~",
    "random": "Random~",
    "randomLoose": "Loose~",
    "randomTight": "Tight~",
    "Total Mean": "Total Mean",
}


def parse_mean_std(value):
    if pd.isna(value):
        return "", ""
    text = str(value).strip()
    match = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\(([-+]?\d+(?:\.\d+)?)\)\s*$", text)
    if match:
        return match.group(1), match.group(2)
    return text, ""


def format_mean_std_latex(value, force_std=False):
    mean, std = parse_mean_std(value)
    if not mean:
        return ""
    if std:
        return f"${mean}$ {{\\tiny $\\pm{std}$}}"
    if force_std:
        return f"${mean}$ {{\\tiny $\\pm0.00$}}"
    return f"${mean}$"


def format_speed(value):
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return f"{str(value).strip()}x"


def format_mean_only(value):
    mean, std = parse_mean_std(value)
    if not mean:
        return ""
    return f"${mean}$"


def format_percent(value):
    mean, _ = parse_mean_std(value)
    if not mean:
        return ""
    return f"${mean}$\\%"


def build_row(dataset, exp1_row, exp2_row):
    ds_label = DATASET_LABELS.get(dataset, dataset)

    left_time = format_mean_std_latex(exp1_row.get("Time (s)", ""), force_std=True)
    left_score = format_mean_std_latex(exp1_row.get("Score", ""), force_std=True)
    left_speed = format_speed(exp1_row.get("Speedup", ""))
    left_drop = format_percent(exp1_row.get("Score Delta (%)", ""))

    right_time = format_mean_std_latex(exp2_row.get("Time (s)", ""), force_std=True)
    right_score = format_mean_std_latex(exp2_row.get("Score", ""), force_std=True)
    right_speed = format_speed(exp2_row.get("Speedup", ""))
    right_drop = format_percent(exp2_row.get("Score Delta (%)", ""))

    return (
        f"{ds_label} & {left_time} & {left_score} & {left_speed} & {left_drop}"
        f" & & {right_time} & {right_score} & {right_speed} & {right_drop} \\\\"
    )


def main():
    exp1_df = pd.read_csv(EXP1_CSV)
    exp2_df = pd.read_csv(EXP2_CSV)

    exp1_map = exp1_df.set_index("Dataset").to_dict("index")
    exp2_map = exp2_df.set_index("Dataset").to_dict("index")

    rows = []
    for dataset in DATASET_ORDER:
        if dataset in exp1_map and dataset in exp2_map:
            rows.append(build_row(dataset, exp1_map[dataset], exp2_map[dataset]))

    total_row = None
    if "Total Mean" in exp1_map and "Total Mean" in exp2_map:
        total_row = build_row("Total Mean", exp1_map["Total Mean"], exp2_map["Total Mean"])

    lines = [
        "{",
        r"\setlength{\tabcolsep}{2pt}",
        r"\scriptsize",
        "",
        r"\begin{tabular}{lrrrrcrrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{{\bf Dataset-Specific Tuning}} & ~ & \multicolumn{4}{c}{{\bf Cross-Dataset Generalization}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-10}",
        r" &  &  & \multicolumn{2}{c}{Weight-Baseline} &  & & & \multicolumn{2}{c}{Weight-Baseline} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){9-10}",
        r"{\bf Dataset} & \multicolumn{1}{c}{{\bf Time (s)}} & \multicolumn{1}{c}{{\bf Score}} & \multicolumn{1}{c}{{\bf Speed$\uparrow$}} & \multicolumn{1}{c}{{\bf Score$\downarrow$}} & & \multicolumn{1}{c}{{\bf Time (s)}} & \multicolumn{1}{c}{{\bf Score}} & \multicolumn{1}{c}{{\bf Speed$\uparrow$}} & \multicolumn{1}{c}{{\bf Score$\downarrow$}} \\",
        r"\midrule",
    ]

    lines.extend(rows)

    if total_row:
        lines.append(r"\midrule")
        lines.append(total_row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "}",
    ])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_TEX)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Combined table saved to {out_path}")


if __name__ == "__main__":
    main()
