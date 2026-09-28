"""Select the CLARA observations taken within half an hour of local midnight."""

import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, subset_data_path, subset_results_path, show_or_save

import matplotlib.pyplot as plt
import pandas as pd

from read_in.midnight_subset.midnight import (
    CLARA_LATITUDE_COLUMN,
    CLARA_LOCAL_TIME_COLUMN,
    CLARA_LONGITUDE_COLUMN,
    CLARA_RADIANCE_COLUMN,
    CLARA_TIME_COLUMN,
    SUBSET,
    WINDOW_HOURS,
    YEAR,
    plan_requests,
    select_midnight,
)


df = pd.read_pickle(data_path("CLARA.pkl"))
df_midnight = select_midnight(df)

out = subset_data_path("CLARA_matched.pkl", subset=SUBSET)
out.parent.mkdir(parents=True, exist_ok=True)
df_midnight.to_pickle(out)
print(
    f"saved {len(df_midnight):,} CLARA rows within {WINDOW_HOURS:g}h of local "
    f"midnight in {YEAR} -> {out}"
)

per_month = df_midnight[CLARA_TIME_COLUMN].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
print("Observations per month:")
print(per_month.to_string())
print(f"Distinct days: {df_midnight[CLARA_TIME_COLUMN].dt.normalize().nunique()}")
print(df_midnight[CLARA_RADIANCE_COLUMN].describe())

# How much ERA5 this selection implies, before anything is downloaded.
_, requests = plan_requests(df_midnight)
print(
    f"ERA5 requests: {len(requests)} "
    f"({int(requests['n_items'].sum()):,} CDS items, "
    f"{int(requests['days'].apply(len).sum()):,} request-days)"
)

plt.figure(figsize=(8, 6))
plt.scatter(
    df_midnight[CLARA_LONGITUDE_COLUMN],
    df_midnight[CLARA_LATITUDE_COLUMN],
    c=df_midnight[CLARA_RADIANCE_COLUMN],
    s=8,
    cmap="viridis",
    alpha=0.7,
)
plt.colorbar(label=CLARA_RADIANCE_COLUMN)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title(f"CLARA locations at local midnight - {YEAR}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/clara_midnight_locations.png", subset=SUBSET))

# The selection wraps around 24h, so the histogram should show two spikes, one
# just before midnight and one just after, and nothing in between.
plt.figure(figsize=(8, 4))
plt.hist(df_midnight[CLARA_LOCAL_TIME_COLUMN], bins=60, range=(0, 24), color="#2563eb")
plt.xlabel(CLARA_LOCAL_TIME_COLUMN)
plt.ylabel("Observations")
plt.title(f"CLARA local time of the selected observations - {YEAR}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
show_or_save(plt, subset_results_path("read_in/clara_midnight_local_time.png", subset=SUBSET))
