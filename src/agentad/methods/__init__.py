"""Neural time-series anomaly-detection methods.

Install the ``methods`` extra before importing this package::

    uv sync --extra methods
"""

from .CrossAD import CrossAD, CrossADConfig, CrossADOutput
from .DADA import DADA, DADAConfig, DADAOutput
from .KanAD import KANAD, KANADConfig
from .PaAno import PaAno, PaAnoConfig, PaAnoLoss, PatchEncoder
from .ScatterAD import ScatterAD, ScatterADConfig, ScatterADLoss, ScatterADOutput

METHODS = {
    "CrossAD": CrossAD,
    "DADA": DADA,
    "KAN-AD": KANAD,
    "PaAno": PaAno,
    "ScatterAD": ScatterAD,
}

__all__ = [
    "METHODS",
    "CrossAD",
    "CrossADConfig",
    "CrossADOutput",
    "DADA",
    "DADAConfig",
    "DADAOutput",
    "KANAD",
    "KANADConfig",
    "PaAno",
    "PaAnoConfig",
    "PaAnoLoss",
    "PatchEncoder",
    "ScatterAD",
    "ScatterADConfig",
    "ScatterADLoss",
    "ScatterADOutput",
]
