import zipfile
import numpy as np
import pandas as pd
from agentad.series import SeriesData, write

features = pd.DataFrame(
    {"name": ["temp"], "unit": ["degC"]}, index=pd.Index(["temp"], name="feature")
)
labels = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1]})
sdata = SeriesData(
    features=features,
    labels=labels,
    data=np.arange(8, dtype=np.float32),
    timestamps=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
    offsets=np.array([0, 3, 8], dtype=np.int64),
    ids=["s1", "s2"],
)
write(sdata, "tmp/demo.zarr.zip", create_parents=True)
with zipfile.ZipFile("tmp/demo.zarr.zip") as zf:
    for info in zf.infolist():
        print(f"{info.file_size:>8}  {info.filename}")
