"""Interactive WebUI for ``SeriesData`` collections and contract-based CSV.

Typical use::

    from agentad import read
    from agentad.visualize import serve

    serve(read("data.zarr.zip"), open_browser=True)

The collection may also be picked in the browser: started without one
(``python -m agentad.visualize``) the WebUI offers a directory tree of the
current directory from which any package can be opened. The server binds to
loopback by default and performs all slicing, normalization and downsampling
lazily for the selected series and window.
"""

from ._normalize import (
    NORMALIZATIONS,
    Normalization,
    NormalizationScope,
    normalize,
)
from ._csv import CSV_FORMAT, read_csv
from ._server import WebUI, serve

__all__ = [
    "NORMALIZATIONS",
    "CSV_FORMAT",
    "Normalization",
    "NormalizationScope",
    "WebUI",
    "normalize",
    "read_csv",
    "serve",
]
