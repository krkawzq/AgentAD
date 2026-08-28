"""CARLA exports."""

from .augmentation import inject_anomalies
from .config import CARLAConfig
from .model import CARLA, CARLALoss, CARLAOutput

__all__ = ["CARLA", "CARLAConfig", "CARLALoss", "CARLAOutput", "inject_anomalies"]
