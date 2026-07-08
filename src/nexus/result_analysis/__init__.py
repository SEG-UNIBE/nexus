"""Result analysis scripts and shared helpers."""

from __future__ import annotations

# Re-export for convenience, but analysis scripts should import from
# `nexus.result_analysis.common` directly.
from .common import configure_plotting, pretty_dataset_name

__all__ = ["configure_plotting", "pretty_dataset_name"]
