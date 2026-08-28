"""Interactive, dependency-free WebUI for ``SeriesData`` collections.

Typical use::

    from agentad import read
    from agentad.visualize import serve

    serve(read("data.zarr.zip"))
"""

from ._normalize import NORMALIZATIONS, normalize
from ._server import WebUI, serve

__all__ = ["NORMALIZATIONS", "WebUI", "normalize", "serve"]

