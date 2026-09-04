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
        "zero_degree_level"
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

