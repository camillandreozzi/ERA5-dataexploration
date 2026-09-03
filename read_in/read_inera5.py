import matplotlib.pyplot as plt
import cfgrib


grib_path = "data/ERA5_matched.grib"

# GRIB files can contain multiple groups, so open_datasets is safer
datasets = cfgrib.open_datasets(grib_path)

print(f"Found {len(datasets)} dataset group(s)")

for i, ds in enumerate(datasets):
    print(f"\nDataset {i}")
    print(ds)
    print("Variables:", list(ds.data_vars))

var_name = "tp"
ds = next(ds for ds in datasets if var_name in ds.data_vars)

# Select one non-empty precipitation map
da = ds["tp"].isel(time=0, step=-1)

# Convert metres to millimetres
da = da * 1000
da.attrs["units"] = "mm"

print("Min:", float(da.min()))
print("Max:", float(da.max()))

plt.figure(figsize=(10, 6))
da.plot(cmap="Blues")
plt.title("Total precipitation from ERA5")
plt.tight_layout()
plt.show()
