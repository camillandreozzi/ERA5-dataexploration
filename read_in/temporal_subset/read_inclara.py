import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, subset_data_path, subset_results_path, show_or_save

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_pickle(data_path("CLARA.pkl"))

df["TimeJD"] = pd.to_datetime(df["TimeJD"])


print(df.head())
print(df.info())

print(df['CLARA_radiance'].unique())

df_dec2020 = df[
    (df["TimeJD"] >= "2020-12-01") &
    (df["TimeJD"] <= "2020-12-05")
]
df_dec2020.to_pickle(subset_data_path("CLARA_matched.pkl", subset="temporal_subset"))

df_plot = df_dec2020[["TimeJD", "CLARA_radiance", "CLARA_fov_longitude", "CLARA_fov_latitude", "CLARA_local_time"]].dropna()
df_plot = df_plot.sort_values("TimeJD")
# df_plot = df_plot[df_plot["CLARA_radiance"] > 0]
plt.figure(figsize=(12, 5))
plt.plot(df_plot["TimeJD"], df_plot["CLARA_radiance"], linewidth=1)
# plt.yscale("log")
plt.xlabel("TimeJD")
plt.ylabel("CLARA_radiance (log scale)")
plt.title("CLARA Radiance Time Series - December 2020")
plt.grid(True, alpha=0.3)
plt.tight_layout()


print(df_dec2020["CLARA_radiance"].describe())

show_or_save(plt, subset_results_path("read_in/clara_temporal_01.png", subset="temporal_subset"))

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
plt.title("CLARA Locations - December 2020")
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/clara_temporal_02.png", subset="temporal_subset"))
