import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, subset_data_path, subset_results_path, show_or_save

import pandas as pd
import matplotlib.pyplot as plt


SUBSET = "spatial_subset"

# Must match AREA in data_fetch.py: [north, west, south, east] (Italy).
AREA = [47.1153931748, 6.7499552751, 36.619987291, 18.4802470232]
NORTH, WEST, SOUTH, EAST = AREA
YEAR = 2020


df = pd.read_pickle(data_path("CLARA.pkl"))

df["TimeJD"] = pd.to_datetime(df["TimeJD"])


print(df.head())
print(df.info())

print(df['CLARA_radiance'].unique())

df_2020_it = df[
    (df["TimeJD"].dt.year == YEAR) &
    (df["CLARA_fov_longitude"] >= WEST) &
    (df["CLARA_fov_longitude"] <= EAST) &
    (df["CLARA_fov_latitude"] >= SOUTH) &
    (df["CLARA_fov_latitude"] <= NORTH)
]
out = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
out.parent.mkdir(parents=True, exist_ok=True)
df_2020_it.to_pickle(out)
print(f"saved {len(df_2020_it):,} CLARA rows -> {out}")

# Months with no observations are simply absent from the modelling data.
per_month = df_2020_it["TimeJD"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
print("CLARA observations per month:")
print(per_month.to_string())

df_plot = df_2020_it[["TimeJD", "CLARA_radiance", "CLARA_fov_longitude", "CLARA_fov_latitude", "CLARA_local_time"]].dropna()
df_plot = df_plot.sort_values("TimeJD")
plt.figure(figsize=(12, 5))
plt.plot(df_plot["TimeJD"], df_plot["CLARA_radiance"], linewidth=1, marker=".")
plt.xlabel("TimeJD")
plt.ylabel("CLARA_radiance")
plt.title(f"CLARA Radiance Time Series - Italy {YEAR}")
plt.grid(True, alpha=0.3)
plt.tight_layout()


print(df_2020_it["CLARA_radiance"].describe())

show_or_save(plt, subset_results_path("read_in/clara_spatial_01.png", subset=SUBSET))

plt.figure(figsize=(8, 6))

plt.scatter(
    df_plot["CLARA_fov_longitude"],
    df_plot["CLARA_fov_latitude"],
    c=df_plot["CLARA_radiance"],
    s=8,
    cmap="viridis",
    alpha=0.7
)

plt.colorbar(label="CLARA_radiance")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"CLARA Locations - Italy {YEAR}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/clara_spatial_02.png", subset=SUBSET))
