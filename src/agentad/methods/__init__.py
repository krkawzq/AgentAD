"""Neural time-series anomaly-detection methods.

Install the project environment before importing this package::

    uv sync
"""

from .AERCA import AERCA, AERCAConfig, AERCALoss, AERCAOutput
from .CARLA import CARLA, CARLAConfig, CARLALoss, CARLAOutput, inject_anomalies
from .CrossAD import CrossAD, CrossADConfig, CrossADOutput
from .DADA import DADA, DADAConfig, DADAOutput
from .KanAD import KANAD, KANADConfig
from .Left import Left, LeftConfig, LeftLoss, LeftOutput
from .PaAno import PaAno, PaAnoConfig, PaAnoLoss, PatchEncoder
from .ScatterAD import ScatterAD, ScatterADConfig, ScatterADLoss, ScatterADOutput

METHODS = {
    "AERCA": AERCA,
    "CARLA": CARLA,
    "CrossAD": CrossAD,
    "DADA": DADA,
    "KAN-AD": KANAD,
    "Left": Left,
    "PaAno": PaAno,
    "ScatterAD": ScatterAD,
}

__all__ = [
    "METHODS",
    "AERCA",
    "AERCAConfig",
    "AERCALoss",
    "AERCAOutput",
    "CARLA",
    "CARLAConfig",
    "CARLALoss",
    "CARLAOutput",
    "CrossAD",
    "CrossADConfig",
    "CrossADOutput",
    "DADA",
    "DADAConfig",
    "DADAOutput",
    "KANAD",
    "KANADConfig",
    "Left",
    "LeftConfig",
    "LeftLoss",
    "LeftOutput",
    "PaAno",
    "PaAnoConfig",
    "PaAnoLoss",
    "PatchEncoder",
    "ScatterAD",
    "ScatterADConfig",
    "ScatterADLoss",
    "ScatterADOutput",
    "inject_anomalies",
]
