"""Shared definitions for the midnight subset (global, 2020, local midnight).

Importing this module has no side effects: read_inclara.py, data_fetch.py and
data_preprocessing.py all need the same selection rule and the same split of
the selected observations into CDS requests, and the fetch must not start a
download just because another script imported it.

"Midnight" here is CLARA's own ``CLARA_local_time`` column within ±30 minutes of
00:00. Note that near midnight that column runs about 1.3 h ahead of solar time
at the footprint (``UTC + CLARA_fov_longitude / 15``), so the matched ERA5 rows
sit at roughly 22-23 h solar time. See the README.

Unlike the other two subsets, only the ERA5 hours and areas where CLARA actually
observed near midnight are downloaded, which is why the requests are built from
the CLARA selection rather than from a fixed box and time range.
"""

import numpy as np
import pandas as pd


SUBSET = "midnight_subset"

YEAR = 2020

# Half-width of the accepted window around local midnight, in hours.
WINDOW_HOURS = 0.5

# Degrees added around the observed bounding box of a request, so that an
# observation near the edge still has its nearest ERA5 grid cell inside the box.
AREA_PADDING_DEGREES = 1.0

CLARA_TIME_COLUMN = "TimeJD"
CLARA_RADIANCE_COLUMN = "CLARA_radiance"
CLARA_LOCAL_TIME_COLUMN = "CLARA_local_time"
CLARA_LATITUDE_COLUMN = "CLARA_fov_latitude"
CLARA_LONGITUDE_COLUMN = "CLARA_fov_longitude"

HOUR_COLUMN = "hour"
REQUEST_COLUMN = "request_id"

# Same 38 variables as read_in/spatial_subset/data_fetch.py. That module cannot
# be imported for them, because importing it starts the download.
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


def hours_from_midnight(local_time):
    """Distance to the nearest midnight, in hours, for a local clock time."""
    local_time = pd.to_numeric(local_time, errors="coerce")
    return np.minimum(local_time, 24.0 - local_time)


def select_midnight(clara, year=YEAR, window_hours=WINDOW_HOURS):
    """The CLARA rows of one year observed within ``window_hours`` of midnight."""
    missing = [
        column
        for column in (CLARA_TIME_COLUMN, CLARA_RADIANCE_COLUMN, CLARA_LOCAL_TIME_COLUMN)
        if column not in clara.columns
    ]
    if missing:
        raise ValueError(f"CLARA input is missing columns: {missing}")

    selected = clara.copy()
    selected[CLARA_TIME_COLUMN] = pd.to_datetime(selected[CLARA_TIME_COLUMN])

    keep = (
        (selected[CLARA_TIME_COLUMN].dt.year == year)
        & selected[CLARA_RADIANCE_COLUMN].notna()
        & (hours_from_midnight(selected[CLARA_LOCAL_TIME_COLUMN]) <= window_hours)
    )
    return selected.loc[keep].copy()


def _bounding_box(longitudes, latitudes, padding=AREA_PADDING_DEGREES):
    """CDS ``area`` = [north, west, south, east], padded and clipped."""
    north = min(90.0, float(np.max(latitudes)) + padding)
    south = max(-90.0, float(np.min(latitudes)) - padding)
    west = max(-180.0, float(np.min(longitudes)) - padding)
    east = min(180.0, float(np.max(longitudes)) + padding)
    return [north, west, south, east]


def plan_requests(clara):
    """Split the selected observations into one CDS request per (month, UTC hour).

    ERA5 is hourly and CLARA is matched on the floored UTC hour, the same key
    ``aggregate_clara_hourly`` uses, so every observation of one request shares
    a single ERA5 timestep. The observations of one hour lie in a narrow band of
    longitudes, which keeps each request's area small.

    Returns ``(observations, requests)``: the input rows with ``hour`` and
    ``request_id`` columns added, and one row per request describing what to
    download. A request whose observations straddle the antimeridian is split
    into a west and an east part, because a CDS area cannot wrap around it.
    """
    observations = clara.copy()
    observations[CLARA_TIME_COLUMN] = pd.to_datetime(observations[CLARA_TIME_COLUMN])
    observations[HOUR_COLUMN] = observations[CLARA_TIME_COLUMN].dt.floor("h")
    observations[REQUEST_COLUMN] = pd.NA

    requests = []

    for (year, month, hour_of_day), group in observations.groupby(
        [
            observations[HOUR_COLUMN].dt.year,
            observations[HOUR_COLUMN].dt.month,
            observations[HOUR_COLUMN].dt.hour,
        ],
        sort=True,
    ):
        longitudes = group[CLARA_LONGITUDE_COLUMN]
        straddles_antimeridian = float(longitudes.max() - longitudes.min()) > 180.0

        if straddles_antimeridian:
            parts = [
                ("w", group[longitudes < 0]),
                ("e", group[longitudes >= 0]),
            ]
        else:
            parts = [("", group)]

        for suffix, part in parts:
            if part.empty:
                continue

            request_id = f"{year}{month:02d}_{hour_of_day:02d}{suffix}"
            north, west, south, east = _bounding_box(
                part[CLARA_LONGITUDE_COLUMN], part[CLARA_LATITUDE_COLUMN]
            )
            days = sorted(int(day) for day in part[HOUR_COLUMN].dt.day.unique())

            observations.loc[part.index, REQUEST_COLUMN] = request_id
            requests.append(
                {
                    REQUEST_COLUMN: request_id,
                    "filename": f"ERA5_midnight_{request_id}.grib",
                    "year": year,
                    "month": month,
                    "hour_of_day": hour_of_day,
                    "days": days,
                    "north": north,
                    "west": west,
                    "south": south,
                    "east": east,
                    "n_observations": len(part),
                    "n_items": len(VARIABLES) * len(days),
                }
            )

    if observations[REQUEST_COLUMN].isna().any():
        raise ValueError("Some observations were not assigned to a request.")

    return observations, pd.DataFrame(requests)
