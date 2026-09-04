import matplotlib.pyplot as plt
import cfgrib


grib_path = "data/ERA5_matched.grib"

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
plt.show()
