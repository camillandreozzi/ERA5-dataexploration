# ERA5-CLARA P-O-C

Proof of concept on two subsets of ERA5, each with its own scripts in
`read_in/<subset>/` and its own `data/<subset>/` and `results/<subset>/` folders:

| subset            | area                 | time                     | ERA5 files                 |
|-------------------|----------------------|--------------------------|----------------------------|
| `spatial_subset`  | Italy box            | all of 2020              | `ERA5_matched_2020MM.grib` |
| `temporal_subset` | global               | 2020-12-01 to 2020-12-05 | `ERA5_matched.grib`        |

The exploratory and modelling scripts are shared. They use whichever subset
`ERA5_SUBSET` names (default `spatial_subset`):

    ERA5_SUBSET=temporal_subset python3 modelling/benchmark_rf.py

The spatial subset is fetched and processed on Euler; see `euler/README.md`.

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
count surviving each step.

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
