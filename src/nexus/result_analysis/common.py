from __future__ import annotations


_PRETTY_DATASET_NAMES = {
    # canonical singular
    "hospital": "Hospital",
    "warehouse": "Warehouse",
    "random": "Random",
    "randomloose": "Loose",
    "randomtight": "Tight",
    "apogames": "Apo-Games",
    # common plural forms used in this repo
    "hospitals": "Hospital",
    "warehouses": "Warehouse",
}


def pretty_dataset_name(name: object) -> str:
    """Map internal dataset ids to paper-friendly display names.

    Keeps unknown dataset names unchanged.
    """

    if name is None:
        return ""

    raw = str(name).strip()
    if not raw:
        return raw

    if raw.lower() == "total mean":
        return "Total Mean"

    return _PRETTY_DATASET_NAMES.get(raw.lower(), raw)


def configure_plotting(
    *,
    font_size: int = 22,
    axes_titlesize: int = 28,
    axes_labelsize: int = 22,
    xtick_labelsize: int = 22,
    ytick_labelsize: int = 22,
    legend_fontsize: int = 24,
    figure_titlesize: int = 28,
    seaborn_style: str = "white",
    grid: bool = False,
) -> None:
    """Apply common plotting style settings for analysis figures.

    Lazy-imports matplotlib/seaborn to keep non-plot scripts lightweight.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": axes_titlesize,
            "axes.labelsize": axes_labelsize,
            "xtick.labelsize": xtick_labelsize,
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize,
            "figure.titlesize": figure_titlesize,
        }
    )

    sns.set_theme(style=seaborn_style, rc=plt.rcParams)
    plt.grid(grid)
