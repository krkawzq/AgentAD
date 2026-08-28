"""AgentAD: tooling for anomaly-detection datasets over time series."""

from .series import SeriesData, SeriesItem, read, write

__all__ = ["SeriesData", "SeriesItem", "read", "write"]
