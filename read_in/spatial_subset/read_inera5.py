import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import subset_data_path, subset_results_path, show_or_save

import warnings

import matplotlib.pyplot as plt
import numpy as np
import cfgrib


SUBSET = "spatial_subset"

# cfgrib caches its message index as a .idx beside each GRIB; the first open of a
# month scans the whole file (~30 minutes), later ones reuse the index.
BACKEND_KWARGS = {}

grib_paths = sorted(subset_data_path(subset=SUBSET).glob("ERA5_matched_2020*.grib"))
if not grib_paths:
    raise FileNotFoundError(f"No monthly ERA5 GRIBs in {subset_data_path(subset=SUBSET)}")

print(f"{len(grib_paths)} monthly GRIB files:")
reference_grid = None
for path in grib_paths:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        datasets = cfgrib.open_datasets(path, backend_kwargs=BACKEND_KWARGS)

    variables = sorted(v for ds in datasets for v in ds.data_vars)
    grid_ds = next(ds for ds in datasets if "latitude" in ds.coords)
    latitude = grid_ds.latitude.values
    longitude = grid_ds.longitude.values
    times = np.concatenate([
        np.ravel(ds["valid_time"].values if "valid_time" in ds.coords else ds["time"].values)
        for ds in datasets
    ])

    print(
        f"  {path.name}: {len(variables)} variables, "
        f"grid {len(latitude)}x{len(longitude)} "
        f"(lat {latitude.min():.2f}..{latitude.max():.2f}, "
        f"lon {longitude.min():.2f}..{longitude.max():.2f}), "
        f"valid times {times.min()} .. {times.max()}"
    )

    grid = (latitude.tolist(), longitude.tolist())
    if reference_grid is None:
        reference_grid = grid
        reference_variables = variables
        print(f"    variables: {', '.join(variables)}")
    else:
        if grid != reference_grid:
            print("    WARNING: lat/lon grid differs from the first month")
        if variables != reference_variables:
            missing = sorted(set(reference_variables) - set(variables))
            extra = sorted(set(variables) - set(reference_variables))
            print(f"    WARNING: variables differ from the first month; missing {missing}, extra {extra}")

    if path == grib_paths[0]:
        # Keep the first month open for the plot below. Re-opening with a
        # filter_by_keys would build a second cfgrib index, i.e. another full
        # scan of the file.
        first_month_datasets = datasets
    else:
        for ds in datasets:
            ds.close()


da = next(ds["sp"] for ds in first_month_datasets if "sp" in ds.data_vars).isel(time=0)

print("Min:", float(da.min()))
print("Max:", float(da.max()))

plt.figure(figsize=(8, 7))
da.plot(cmap="viridis")
plt.title(f"Surface pressure from ERA5 - {grib_paths[0].stem}")
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/era5_spatial_01.png", subset=SUBSET))
