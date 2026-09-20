import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, subset_data_path, subset_results_path, show_or_save

import matplotlib.pyplot as plt
import cfgrib


grib_path = subset_data_path("ERA5_matched.grib", subset="temporal_subset")

ds = cfgrib.open_dataset(
    grib_path,
    backend_kwargs={"filter_by_keys": {"shortName": "sp"}},
)

da = ds["sp"].isel(time=0)

print("Min:", float(da.min()))
print("Max:", float(da.max()))

plt.figure(figsize=(10, 6))
da.plot(cmap="viridis")
plt.title("Surface pressure from ERA5")
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/era5_temporal_01.png", subset="temporal_subset"))
