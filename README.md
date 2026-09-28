# ERA5-CLARA P-O-C

Proof of concept on three subsets of ERA5, each with its own scripts in
`read_in/<subset>/` and its own `data/<subset>/` and `results/<subset>/` folders:

| subset             | area                 | time                          | ERA5 files                    |
|--------------------|----------------------|-------------------------------|-------------------------------|
| `spatial_subset`   | Italy box            | all of 2020                   | `ERA5_matched_2020MM.grib`    |
| `temporal_subset`  | global               | 2020-12-01 to 2020-12-05      | `ERA5_matched.grib`           |
| `midnight_subset`  | global               | 2020, CLARA local midnight    | `ERA5_midnight_2020MM_HH.grib`|

The exploratory and modelling scripts are shared. They use whichever subset
`ERA5_SUBSET` names (default `spatial_subset`):

    ERA5_SUBSET=temporal_subset python3 modelling/benchmark_rf.py

The spatial subset is fetched and processed on Euler; see `euler/README.md`.
The midnight subset is small enough to run on a laptop; see below.

## Step by step (for `<subset>`):
1. read_in/<subset>/read_inclara.py
2. read_in/<subset>/data_fetch.py (area/time decided in read_inclara.py)
3. read_in/<subset>/read_inera5.py
4. exploratory/data_overlap.py
5. read_in/<subset>/data_preprocessing.py
6. exploratory/explore_merged.py, explore_era5.py, explore_clara.py
7. modelling/benchmark_lasso.py, benchmark_rf.py, rf_stkriging.py

Run `python3 exploratory/explore_era5.py` to regenerate ERA5 exploration outputs
in `results/<subset>/exploratory/era5/`. Alongside distributions and overall missingness,
it produces `covariate_missingness_by_hour.png` (variables × UTC hours) and
`covariate_missingness_spatial_01.png`, etc. (12 variable maps per page).
The temporal plot shows the fraction of grid cells missing at each hour; spatial
maps show the fraction of scanned rows missing at each native ERA5 cell over time.
All merged ERA5 rows are included, regardless of CLARA availability. NaN and
infinite values count as missing; zero and negative finite values remain valid.
Raw variables and the existing derived covariates are included.

Exact counts and rates are saved in `covariate_missingness_by_hour.csv` and
`covariate_missingness_by_location.parquet` (one row per variable/location).
These are streamed summaries, not distribution samples. `--max-batches N`
limits the scan and marks the new outputs as partial; denominators then refer
only to scanned rows. Missingness describes the merged dataset: it can reflect
land/ocean masks or hours unavailable after merging, not just gaps in the source
GRIB. Entirely absent rows are not inferred as missing.

## The midnight subset

CLARA observations taken within 30 minutes of local midnight, anywhere on the
globe, over 2020: 1,278 observations on 204 days, which merge into about a
thousand hourly grid cells. 2020 is used because it is the year with the fewest
CLARA gaps (16 missing days, longest gap 6 days; 2021-2023 each have gaps of
98-164 days), and it holds 1,278 of the 1,785 near-midnight observations in the
whole record.

Two things differ from the other subsets.

**Only CLARA-matched rows are kept.** The others keep every ERA5 hour x
longitude x latitude row and attach CLARA where it exists. At local midnight
that would mean 24 global hours a day, hundreds of GB, for about a thousand
usable rows. So `read_in/midnight_subset/midnight.py` builds one CDS request per
(month, UTC hour) covering just the box where CLARA observed in that hour: 103
requests and well under a GB for 2020. `data_fetch.py` skips requests already on
disk, so an interrupted run resumes.

That resumption is per-request, not per-variable: a request already on disk is
skipped whole, so adding a variable to `VARIABLES` does not backfill the files
already fetched. To pick up the four top-of-atmosphere fields, delete the
existing `ERA5_midnight_*.grib` (and their `.idx`) and re-run the fetch.

This also means `rf_stkriging.py --mode predict-grid` has no grid to predict on
(use `--mode validate`), and the ERA5-left exploration — `explore_era5.py`
missingness maps and `data_overlap.py` — does not apply here.

**Midnight means CLARA's own clock.** The selection is `CLARA_local_time` within
±30 min of 00:00. Be aware that near midnight that column runs about 1.3 h ahead
of solar time at the footprint (`UTC + CLARA_fov_longitude / 15`), so the matched
ERA5 rows sit at roughly 22-23 h solar time. Selecting on footprint solar time
instead would pick a largely different set of observations (only 139 in common)
spread over 40 days rather than 204.

Everything runs locally:

    export ERA5_SUBSET=midnight_subset
    python3 read_in/midnight_subset/read_inclara.py        # select + plan requests
    python3 read_in/midnight_subset/data_fetch.py          # 103 requests, resumable
    python3 read_in/midnight_subset/read_inera5.py         # inventory of what arrived
    python3 read_in/midnight_subset/data_preprocessing.py  # -> CLARA_ERA5_merged.parquet
    python3 modelling/benchmark_lasso.py                   # and benchmark_rf.py
    python3 modelling/rf_stkriging.py --mode validate

The fetch waits on the CDS queue, so it is the slow step. To try the chain on one
request first:

    python3 read_in/midnight_subset/data_fetch.py --limit-requests 1
    python3 read_in/midnight_subset/data_preprocessing.py --allow-missing-files

## Full CLARA record

Run `python3 exploratory/explore_clara_full.py` to explore the CLARA time series
in `results/exploratory/clara_full/`. Unlike `explore_clara.py`, it reads
`data/CLARA.pkl` rather than the ERA5-matched subset, so it is not limited to the
proof-of-concept window.

The file spans 2020-01-01 to 2023-12-31, but coverage is dense only until
2021-08; a 10-month gap follows and 2022-2023 is sparse and noisy. The default
window is therefore 2020-01-01 to 2021-12-31, whose last retained observation is
2021-08-19. `--start` and `--end` move the bounds (a bare `--end` date keeps that
whole day); pass `none` to either to use the whole record.

`CLARA.pkl` mixes Earth-view radiances with calibration and off-nominal views:
about 12% of rows are negative, 6% are exactly zero, and a separate cluster sits
near 9e4 (with two values above 1e9). The default filter keeps
`0 < radiance <= 500`, retaining 125,076 of the 159,550 rows in the default
window; `--radiance-min`/`--radiance-max` move the bounds and
`--no-radiance-filter` keeps everything finite. `row_retention.csv` records the
count surviving each step. The merge path applies the same bounds; see
"The radiance filter" under "For the modelling".

De-trending fits a polynomial in years since the first retained timestamp
(`--trend-degree`, default 1) and subtracts it. The fit uses daily means by
default so it is not dominated by sampling density; `--trend-fit observations`
fits individual rows instead. Coefficients, R-squared, and residual moments go to
`trend_fit_summary.csv`, and `clara_full_series_with_trend.png` shows the daily
mean series with the fitted line. Over the default window the slope is about
-5.9 radiance/year with R-squared 0.04, i.e. almost no linear trend; run with
`--end none` and it steepens to -17.3/year with R-squared 0.35, but that slope is
an artefact of the sparse 2022-2023 tail rather than a real decline.

Residuals are then averaged onto daily, weekly, monthly, and yearly grids. Each
granularity gets one figure, `clara_residual_<granularity>.png`, with the residual
series on top and its autocorrelation below. Empty periods stay NaN so data gaps
break the plotted line rather than being interpolated across; `--connect-gaps`
drops them so the line joins straight across instead. That choice affects plotting
only. Autocorrelation always runs on the regular grid with empty periods left as
NaN, so a lag of k still means k whole periods. It is a pairwise-complete Pearson
correlation per lag; lags with fewer than 6 usable pairs are dropped, and the
dashed band is `1.96/sqrt(n_pairs)` at each lag. `--max-lag-daily`,
`--max-lag-weekly`, `--max-lag-monthly`, and `--max-lag-yearly` set the lag
ranges. Per-granularity values are saved in `residual_series_<granularity>.csv`
and `residual_autocorrelation_<granularity>.csv`.

The daily residual autocorrelation decays smoothly from about 0.36 at lag 1 and
crosses zero near lag 60. The monthly one is positive at lag 1, negative through
lags 4-9, and positive again at lags 12-13, the signature of the annual cycle that
de-trending deliberately leaves in. The yearly panel spans only 2 periods in the
default window, too few to support any lag, so its autocorrelation axis reports
that instead of a plot.

## The ERA5 variables

The fetch scripts request 42 single-level variables. The list is duplicated in
three places that must stay in step: `read_in/spatial_subset/data_fetch.py`,
`read_in/midnight_subset/midnight.py`, and the inline `"variable"` list in
`read_in/temporal_subset/data_fetch.py`.

The first four are top-of-atmosphere longwave and matter more than the rest
together:

| variable | why |
|---|---|
| `mean_top_net_long_wave_radiation_flux` | ERA5's own OLR — the same physical quantity CLARA measures |
| `mean_top_net_long_wave_radiation_flux_clear_sky` | cloud radiative effect, by difference with the above |
| `total_column_cloud_ice_water` | cold high cloud |
| `total_column_cloud_liquid_water` | cloud water column |

They were added after a diagnostic showed the original 38 were all surface or
column-integrated fields, with no top-of-atmosphere longwave at all: the only
TOA variable was `toa_incident_solar_radiation`, incoming shortwave, which is
identically zero at local midnight. OLR is emitted from cloud tops and the
upper troposphere, so nothing in the set described the emitting layer. The
symptom was that `era5_hcc`, high cloud cover and the dominant control on OLR,
correlated +0.002 with the target.

The same diagnostic, on the midnight subset, showed where the skill goes:

| validation | R-squared (radiance) |
|---|---|
| random 5-fold (interpolation) | +0.284 |
| grouped by month | +0.160 |
| grouped by latitude band | +0.067 |
| space-time blocked (what the benchmarks report) | +0.052 |

and that the ERA5 covariates carry almost nothing that transfers across
latitude: `era5_skt` has marginal correlation +0.43 with the target but scores
-0.101 alone under latitude-blocked validation, because that correlation is
entirely between-latitude. Dropping `latitude`/`longitude` makes the blocked
score worse (+0.067 to +0.017), so the coordinates are supplying information
the covariates do not.

**These four have not been downloaded yet.** The lists are updated but the
GRIBs on disk predate them, so the models still run on the previous 38 and
`load_model_data` simply skips the absent columns. Re-running the fetch is what
puts them in play.

The modelling allow-lists in `benchmark_rf.py` and `benchmark_lasso.py` name
several spellings of the top-of-atmosphere fields (`era5_avg_tnlwrf`,
`era5_mtnlwrf`, `era5_ttr`, and the clear-sky forms). ECMWF names mean-rate
parameters `avg_*` in recent GRIB — `mean_surface_net_long_wave_radiation_flux`
arrives as `era5_avg_snlwrf` — but which spelling cfgrib emits for the TOA
fields has not been checked against a downloaded file. Absent names are skipped,
so listing all of them costs nothing; once a GRIB arrives, trim the list to the
one that appears.

## For the modelling
benchmark_lasso and benchmark_rf only model the mean function of OLR
in rf_stkriging the attempt is to combine a non-linear mean modelling with a space-time kriging of the residuals

The mean-model pipelines one-hot encode ERA5 low/high vegetation type (`tvl`, `tvh`)
and soil type (`slt`). Vegetation cover fractions, leaf-area indices, and the
land-sea fraction remain continuous. Encoding and imputation are fitted on each
outer training fold; missing categories use a separate code and unseen categories
encode as all zeros. RF importance and Lasso coefficient outputs report individual
encoded categories. RF residual kriging uses the same preprocessing for its mean
model, with unchanged space-time coordinates. ERA5 hourly preprocessing rejects
conflicting static category codes instead of averaging them.

RF, Lasso, and RF residual kriging include decimal-hour `local_time` as a continuous
covariate. They read it from the merged data; older files without this column use
CET clock time derived from UTC `hour` for both training and full-grid prediction.
Existing local times must be finite and in [0, 24).

### The radiance filter

`aggregate_clara_hourly` rejects CLARA observations outside `0 < radiance <= 500`
before the hourly aggregation, so a calibration or off-nominal view never becomes
a grid cell and never counts towards `clara_n_datapoints`. Cells left with no
valid observation are not emitted. Each run prints how many observations were
rejected and why.

Until this was added the filter existed only in `explore_clara_full.py`, so
nothing on the path the models read applied it. That mattered differently per
subset. In `temporal_subset` the bad values are the 151 nonpositive ones (15% of
1,030), which `add_log_radiance` was already discarding downstream — the filter
moves that loss to where it is visible but does not change the training set. In
`midnight_subset` the bad values are the 142 above 500 (11% of 1,278, max
91,913); those are positive, so `add_log_radiance` never caught them and they
were being trained on.

The bounds are named constants in `read_in/temporal_subset/data_preprocessing.py`.
The upper one is an empirical gap rather than a physical limit: valid Earth views
run to about 500 and the next cluster sits near 9e4, so any bound between roughly
600 and 1e4 selects the same rows. The units of `CLARA_radiance` are not
established here, so it has not been checked against a physical bound.

**Known limitation.** The nonpositive observations are discarded, not corrected.
They are not calibration views: their housekeeping and geometry match valid Earth
views, and they sit at a median of -13.7 with quartiles -20.9 to -7.2. That is the
signature of a zero-point offset of order 15, roughly 10% of the median radiance,
which would bias every observation rather than only those pushed below zero.
Discarding removes the visible symptom and leaves the offset in place.

Models retain strictly positive, finite hourly radiance observations and add
`log_radiance = log(clara_radiance_hourly_mean)` as the training target. Zero and
negative observations are excluded because their real logarithms are undefined.
Predictions use `exp()` without bias correction, and errors, metrics, and radiance
plots use the original radiance scale. The baseline is `exp(mean(log_radiance))`
from each training fold. RF residual kriging fits residuals on the log scale and
returns `exp(rf_prediction + kriged_residual)`. Its `kriged_residual_factor` is a
multiplicative correction; `kriging_log_variance` and variogram sill/nugget values
remain on the log scale for diagnostics.
RF residual kriging excludes `era5_sst` and `era5_siconc` from its covariates for
both validation and grid prediction, so their ocean-only coverage does not remove
land cells. The standalone RF and Lasso benchmarks retain their existing covariates.

RF + kriging validation and grid outputs also include approximate uncertainty in
radiance squared. `rf_log_variance` is the sample variance across individual tree
predictions (`ddof=1`), streamed after applying the fitted preprocessor. This is a
tree-disagreement proxy, not calibrated sampling variance of the forest mean;
it is not divided by the number of correlated trees. RF and kriging log errors
are approximated as independent Gaussian variables. With combined log prediction
`mu` and total log variance `v = rf_log_variance + kriging_log_variance`, the
[lognormal variance](https://itl.nist.gov/div898/handbook/apr/section1/apr164.htm)
is `predicted_radiance_variance = exp(2*mu + v) * expm1(v)`.

Radiance-scale contributions sum exactly to that total using the law of total
variance conditional on the RF log prediction: `rf_radiance_variance` is
`Var(E[Y | RF])`, and `kriging_radiance_variance` is `E[Var(Y | RF)]`. Writing
`r = rf_log_variance`, `k = kriging_log_variance`, and `S = exp(2*mu + r + k)`,
these contributions are `S*expm1(r)` and `S*exp(r)*expm1(k)`, respectively.
The map shows predictions, total variance, and these two contributions, with a
common variance color scale. Unknown kriging variance propagates to unknown
total variance. Point predictions still use `exp(mu)` without a mean correction.
Shared training data, in-sample RF residuals, and fitted variogram uncertainty
limit this approximation; it is not a calibrated full predictive distribution.

## Writing new code: inputs and outputs

Never hardcode `"data/..."` or `"results/..."`. Those break on the cluster,
where the files are not next to the code. Start every new script with this
block, copied exactly — it is the same in every folder, at any depth:

```python
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
from paths import data_path, results_path
```

Then:

```python
frame = pd.read_parquet(subset_data_path("CLARA_ERA5_merged.parquet"))   # input
frame.to_csv(subset_results_path("modelling/my_thing/scores.csv"))       # output
```

- `subset_data_path(...)` / `subset_results_path(...)` — anything belonging to
  a subset. They resolve under `data/$ERA5_SUBSET/` and `results/$ERA5_SUBSET/`;
  pass `subset="temporal_subset"` to pin one explicitly.
- `data_path(...)` / `results_path(...)` — files shared by all subsets
  (`CLARA.pkl`, the full-record CLARA exploration).
- The `results_*` helpers create the folder for you.

Both return absolute paths. On your laptop they land in `./data` and
`./results`. On Euler they follow `ERA5_DATA_ROOT` / `ERA5_RESULTS_ROOT` from
`euler/env.sh`. Same code, no edits, both machines.

Quick check that a new script is wired right:

    python -c "import paths; print(paths.DATA_ROOT, paths.RESULTS_ROOT)"
