import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "year": [
        "2020"
    ],
    "month": [
        "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05"
    ],
    "variable": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "instantaneous_10m_wind_gust",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation",
        "convective_precipitation",
        "snow_depth",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "total_column_water_vapour",
        "volumetric_soil_water_layer_1",
        "boundary_layer_height",
        "convective_available_potential_energy"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "area": [90, -180, -20, 180],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("data/ERA5_matched.grib")

