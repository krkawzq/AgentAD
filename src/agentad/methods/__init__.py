"""Neural time-series anomaly-detection methods.

Install the project environment before importing this package::

    uv sync
"""

from .AERCA import AERCA, AERCAConfig, AERCALoss, AERCAOutput
from .AxonAD import AxonAD, AxonADConfig, AxonADLoss, AxonADOutput
from .CARLA import CARLA, CARLAConfig, CARLALoss, CARLAOutput, inject_anomalies
from .CrossAD import CrossAD, CrossADConfig, CrossADLoss, CrossADOutput
from .DADA import DADA, DADAConfig, DADAOutput
from .KanAD import KANAD, KANADConfig, KANADLoss, KANADOutput
from .Left import Left, LeftConfig, LeftLoss, LeftOutput
from .MMPAD import MMPAD, MMPADConfig
from .PaAno import PaAno, PaAnoConfig, PaAnoLoss, PatchEncoder
from .ScatterAD import ScatterAD, ScatterADConfig, ScatterADLoss, ScatterADOutput
from .TimeRCD import TimeRCD, TimeRCDConfig, TimeRCDLoss, TimeRCDOutput
from .TSPulse import (
    TSPulse,
    TSPulseConfig,
    TSPulseFineTune,
    TSPulseLoss,
    TSPulseOutput,
    TSPulseZeroShot,
)
from .XLSTMAD import XLSTMAD, XLSTMADConfig, XLSTMADLoss, XLSTMADOutput

METHODS = {
    "AERCA": AERCA,
    "AxonAD": AxonAD,
    "CARLA": CARLA,
    "CrossAD": CrossAD,
    "DADA": DADA,
    "KAN-AD": KANAD,
    "Left": Left,
    "MMPAD": MMPAD,
    "PaAno": PaAno,
    "ScatterAD": ScatterAD,
    "Time-RCD": TimeRCD,
    "TSPulse-FT": TSPulseFineTune,
    "TSPulse-ZS": TSPulseZeroShot,
    "xLSTMAD": XLSTMAD,
}

__all__ = [
    "METHODS",
    "AERCA",
    "AERCAConfig",
    "AERCALoss",
    "AERCAOutput",
    "AxonAD",
    "AxonADConfig",
    "AxonADLoss",
    "AxonADOutput",
    "CARLA",
    "CARLAConfig",
    "CARLALoss",
    "CARLAOutput",
    "CrossAD",
    "CrossADConfig",
    "CrossADLoss",
    "CrossADOutput",
    "DADA",
    "DADAConfig",
    "DADAOutput",
    "KANAD",
    "KANADConfig",
    "KANADLoss",
    "KANADOutput",
    "Left",
    "LeftConfig",
    "LeftLoss",
    "LeftOutput",
    "MMPAD",
    "MMPADConfig",
    "PaAno",
    "PaAnoConfig",
    "PaAnoLoss",
    "PatchEncoder",
    "ScatterAD",
    "ScatterADConfig",
    "ScatterADLoss",
    "ScatterADOutput",
    "TimeRCD",
    "TimeRCDConfig",
    "TimeRCDLoss",
    "TimeRCDOutput",
    "TSPulse",
    "TSPulseConfig",
    "TSPulseFineTune",
    "TSPulseLoss",
    "TSPulseOutput",
    "TSPulseZeroShot",
    "XLSTMAD",
    "XLSTMADConfig",
    "XLSTMADLoss",
    "XLSTMADOutput",
    "inject_anomalies",
]
