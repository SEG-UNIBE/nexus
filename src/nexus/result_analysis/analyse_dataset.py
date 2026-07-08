from __future__ import annotations

from pathlib import Path
import pandas as pd

from nexus.result_analysis.common import pretty_dataset_name

DATASETS = [
    "warehouses",
    "hospitals",
    "Apogames",
    "random",
    "randomLoose",
    "randomTight",
]

ASSET_DIR = Path("src/assets")
OUT_DIR = Path("output/dataset_summary")
OUT_CSV = OUT_DIR / "dataset_summary.csv"
OUT_TEX = OUT_DIR / "dataset_summary.tex"


def count_attributes(value: object) -> int:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return 0
    return len([part for part in text.split(";") if part.strip()])


def load_dataset(dataset_name: str) -> pd.DataFrame:
    path = ASSET_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(
        path,
        header=None,
        names=["model", "gt_hash", "element", "attributes"],
        dtype={"model": str, "gt_hash": str, "element": str, "attributes": str},
    )
    df["dataset"] = dataset_name
    df["attribute_count"] = df["attributes"].apply(count_attributes)
    return df


def format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def main() -> None:
    frames = []
    for dataset_name in DATASETS:
        frames.append(load_dataset(dataset_name))

    all_df = pd.concat(frames, ignore_index=True)

    summary = (
        all_df.groupby("dataset")
        .agg(
            total_elements=("element", "size"),
            nr_models=("model", pd.Series.nunique),
            avg_nr_attributes=("attribute_count", "mean"),
        )
        .reset_index()
    )
    summary["avg_nr_elements_per_model"] = summary["total_elements"] / summary["nr_models"]

    summary = summary[
        [
            "dataset",
            "nr_models",
            "total_elements",
            "avg_nr_elements_per_model",
            "avg_nr_attributes",
        ]
    ].copy()

    summary["dataset"] = pd.Categorical(summary["dataset"], categories=DATASETS, ordered=True)
    summary = summary.sort_values("dataset").reset_index(drop=True)
    summary["dataset"] = summary["dataset"].astype(str)

    summary_out = summary.copy()
    summary_out["dataset"] = summary_out["dataset"].apply(pretty_dataset_name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(OUT_CSV, index=False)

    table_lines = [
        "{",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\scriptsize",
        "",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{No. Models} & \textbf{Total Elements} & \textbf{Avg. Elements/Model} & \textbf{Avg. Properties} \\",
        r"\midrule",
    ]

    for _, row in summary_out.iterrows():
        table_lines.append(
            f"{row['dataset']} & {int(row['nr_models'])} & {int(row['total_elements'])} & "
            f"{format_float(row['avg_nr_elements_per_model'])} & {format_float(row['avg_nr_attributes'])} \\\\"
        )

    total_row = {
        "dataset": "Mean",
        "total_elements": all_df.shape[0],
        "nr_models": all_df["model"].nunique(),
        "avg_nr_attributes": all_df["attribute_count"].mean(),
        "avg_nr_elements_per_model": all_df.shape[0] / all_df["model"].nunique(),
    }

    table_lines.extend([
        r"\midrule",
        (
            f"{total_row['dataset']} & {int(total_row['nr_models'])} & {int(total_row['total_elements'])} & "
            f"{format_float(total_row['avg_nr_elements_per_model'])} & "
            f"{format_float(total_row['avg_nr_attributes'])} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "}",
    ])

    with open(OUT_TEX, "w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines) + "\n")

    print(f"Summary CSV saved to {OUT_CSV}")
    print(f"Summary LaTeX table saved to {OUT_TEX}")
    print(summary_out.to_string(index=False))


if __name__ == "__main__":
    main()
