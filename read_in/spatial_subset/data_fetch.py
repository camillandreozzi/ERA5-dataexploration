import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, results_path

import os

import cdsapi

# Useful Link: https://confluence.ecmwf.int/pages/viewpage.action?pageId=402639006#heading-Parameterlistings

# CDS bills by "items" = variable x level x timestep. The `area` sub-setting is
# NOT taken into account, so cropping the box shrinks the download but not the
# cost. 38 variables x 24 hours x 182 days = 165,984 items, over the 120,000
# limit for reanalysis-era5-single-levels -> one request per month instead.

DATASET = "reanalysis-era5-single-levels"
OUT_DIR = data_path("spatial_subset")

VARIABLES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux_clear_sky",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux_clear_sky",
    "surface_net_solar_radiation",
    "toa_incident_solar_radiation",
    "high_vegetation_cover",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "low_vegetation_cover",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "skin_temperature",
    "high_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "lake_cover",
    "lake_ice_temperature",
    "lake_mix_layer_temperature",
    "snow_albedo",
    "temperature_of_snow_layer",
    "soil_temperature_level_1",
    "soil_type",
    "volumetric_soil_water_layer_1",
    "forecast_albedo",
    "land_sea_mask",
    "sea_ice_cover",
    "total_column_ozone",
    "total_column_water",
    "total_column_water_vapour",
    "zero_degree_level",
]

DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]

YEAR = "2020"
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Italy: 47.1153931748, 6.7499552751, 36.619987291, 18.4802470232
AREA = [47.1153931748, 6.7499552751, 36.619987291, 18.4802470232]

client = cdsapi.Client()
os.makedirs(OUT_DIR, exist_ok=True)

for month in MONTHS:
    target = os.path.join(OUT_DIR, f"ERA5_matched_{YEAR}{month}.grib")
    if os.path.exists(target):
        print(f"skipping {target}, already downloaded")
        continue

    request = {
        "product_type": ["reanalysis"],
        "year": [YEAR],
        "month": [month],
        "day": DAYS,
        "variable": VARIABLES,
        "time": TIMES,
        "area": AREA,
        "data_format": "grib",
        "download_format": "unarchived",
    }

    print(f"requesting {YEAR}-{month} ({len(VARIABLES) * len(TIMES) * len(DAYS)} items max)")
    client.retrieve(DATASET, request).download(target)
