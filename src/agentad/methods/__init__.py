"""Neural time-series anomaly-detection methods.

Install the project environment before importing this package::

    uv sync
"""

from .AERCA import AERCA, AERCAConfig, AERCALightningModule, AERCALoss, AERCAOutput
from .AxonAD import AxonAD, AxonADConfig, AxonADLightningModule, AxonADLoss, AxonADOutput
from .CARLA import (
    CARLA,
    CARLAClassificationLightningModule,
    CARLAConfig,
    CARLALoss,
    CARLAOutput,
    CARLAPretextLightningModule,
    inject_anomalies,
)
from .CrossAD import CrossAD, CrossADConfig, CrossADLightningModule, CrossADLoss, CrossADOutput
from .DADA import DADA, DADAConfig, DADALightningModule, DADAOutput
from .KanAD import KANAD, KANADConfig, KANADLightningModule, KANADLoss, KANADOutput
from .Left import Left, LeftConfig, LeftLightningModule, LeftLoss, LeftOutput
from .MMPAD import MMPAD, MMPADConfig, MMPADLightningModule
from .PaAno import PaAno, PaAnoConfig, PaAnoLightningModule, PaAnoLoss, PatchEncoder
from .ScatterAD import (
    ScatterAD,
    ScatterADConfig,
    ScatterADLightningModule,
    ScatterADLoss,
    ScatterADOutput,
)
from .TimeRCD import TimeRCD, TimeRCDConfig, TimeRCDLightningModule, TimeRCDLoss, TimeRCDOutput
from .TSPulse import (
    TSPulse,
    TSPulseConfig,
    TSPulseFineTune,
    TSPulseLightningModule,
    TSPulseLoss,
    TSPulseOutput,
    TSPulseZeroShot,
)
from .XLSTMAD import XLSTMAD, XLSTMADConfig, XLSTMADLightningModule, XLSTMADLoss, XLSTMADOutput

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
    "AERCALightningModule",
    "AERCALoss",
    "AERCAOutput",
    "AxonAD",
    "AxonADConfig",
    "AxonADLightningModule",
    "AxonADLoss",
    "AxonADOutput",
    "CARLA",
    "CARLAConfig",
    "CARLAClassificationLightningModule",
    "CARLALoss",
    "CARLAOutput",
    "CARLAPretextLightningModule",
    "CrossAD",
    "CrossADConfig",
    "CrossADLightningModule",
    "CrossADLoss",
    "CrossADOutput",
    "DADA",
    "DADAConfig",
    "DADALightningModule",
    "DADAOutput",
    "KANAD",
    "KANADConfig",
    "KANADLightningModule",
    "KANADLoss",
    "KANADOutput",
    "Left",
    "LeftConfig",
    "LeftLightningModule",
    "LeftLoss",
    "LeftOutput",
    "MMPAD",
    "MMPADConfig",
    "MMPADLightningModule",
    "PaAno",
    "PaAnoConfig",
    "PaAnoLightningModule",
    "PaAnoLoss",
    "PatchEncoder",
    "ScatterAD",
    "ScatterADConfig",
    "ScatterADLightningModule",
    "ScatterADLoss",
    "ScatterADOutput",
    "TimeRCD",
    "TimeRCDConfig",
    "TimeRCDLightningModule",
    "TimeRCDLoss",
    "TimeRCDOutput",
    "TSPulse",
    "TSPulseConfig",
    "TSPulseFineTune",
    "TSPulseLightningModule",
    "TSPulseLoss",
    "TSPulseOutput",
    "TSPulseZeroShot",
    "XLSTMAD",
    "XLSTMADConfig",
    "XLSTMADLightningModule",
    "XLSTMADLoss",
    "XLSTMADOutput",
    "inject_anomalies",
]
