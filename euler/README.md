# Running on Euler

Everything heavy (CDS download + modelling) runs on the cluster. The laptop
holds the code and receives results.

## Storage layout

Large files never live inside the repo checkout. `paths.py` resolves them from
two environment variables, set in `env.sh`:

| variable            | default on Euler              |
|---------------------|-------------------------------|
| `ERA5_DATA_ROOT`    | `$HOME/era5-store/data`       |
| `ERA5_RESULTS_ROOT` | `$HOME/era5-store/results`    |

With neither set (i.e. on the laptop) both fall back to `./data` and
`./results`, so local runs are unchanged.

Inside both roots each data subset has its own folder, picked by
`ERA5_SUBSET` (set to `spatial_subset` in `env.sh`):

    data/CLARA.pkl                                  # shared, pushed from the laptop
    data/spatial_subset/ERA5_matched_2020MM.grib    # fetched on Euler (12 files)
    data/spatial_subset/CLARA_matched.pkl           # built by preprocess.sbatch
    data/spatial_subset/CLARA_ERA5_merged.parquet   # built by preprocess.sbatch
    results/spatial_subset/{read_in,preprocessing,exploratory,modelling}/

**Quota warning.** `/cluster/home` is backed up but capped at roughly 45 GB and
450k inodes. The twelve spatial-subset GRIBs plus the merged parquet (about
17.5M rows) take several GB. cfgrib writes one small `.idx` beside each GRIB,
which is 12 extra files here -- irrelevant for the inode quota, and worth it:
building that index scans every message in the file (~30 min per month), and
without it on disk every single open pays that cost again. Run `lquota` before a big fetch. To move
the data, change `ERA5_DATA_ROOT` in `env.sh` — no Python changes needed:

    /cluster/work/<group>/$USER/era5/data   # permanent, needs a group allocation
    $SCRATCH/era5/data                      # 2.5 TB but PURGED after 15 days

## First-time setup

### Step 1 -- ON YOUR MAC (prompt looks like `camillandreozzi@Mac`)

Run these from a LOCAL terminal, not from an ssh session. Generating the key
on Euler is the common mistake: a key made there authenticates Euler outbound
to other machines, which is not what we need.

    ssh-keygen -t ed25519                       # only if ~/.ssh/id_ed25519 is missing
    ssh-copy-id candreozzi@euler.ethz.ch        # needs ETH network or VPN
    scp ~/.cdsapirc candreozzi@euler.ethz.ch:~/.cdsapirc
    bash euler/push_code.sh

Check you are on the right machine first: `hostname` should NOT start with
`eu-login`. The key is only for passwordless rsync -- your normal
password/2FA login keeps working regardless.

### Step 2 -- ON EULER (prompt looks like `andreozzi@eu-login-42`)

    ssh candreozzi@euler.ethz.ch
    cd ~/ERA5-dataexploration
    bash euler/setup.sh

### Step 3 -- ON YOUR MAC, once

CLARA exists only on your laptop; Euler cannot fetch it from anywhere.

    bash euler/push_data.sh                     # sends data/CLARA.pkl only

ERA5 is not pushed -- Euler downloads that from Copernicus directly.

## Day to day

    bash euler/push_code.sh                     # laptop -> cluster (code only)

Note `push_code.sh` runs rsync with `--delete`: the cluster copy is made to
match your laptop exactly. Edit code on the LAPTOP only -- anything you change
directly on Euler is erased by the next push. (`push_data.sh` has no
`--delete`, so cluster data is never wiped this way.)

    # on Euler, in order (each step needs the previous one's output)
    sbatch euler/fetch_era5.sbatch              # 12 monthly GRIBs, Italy 2020
    sbatch euler/preprocess.sbatch              # CLARA subset + merged parquet
    sbatch euler/run_modelling.sbatch modelling/benchmark_lasso.py
    sbatch euler/run_modelling.sbatch modelling/benchmark_rf.py
    sbatch euler/run_modelling.sbatch modelling/rf_stkriging.py --mode both
    squeue -u $USER                             # watch
    tail -f logs/era5_preprocess_*.out

To chain them without waiting, use `--dependency`:

    pre=$(sbatch --parsable euler/preprocess.sbatch)
    for s in benchmark_lasso benchmark_rf rf_stkriging; do
        sbatch --dependency=afterok:$pre euler/run_modelling.sbatch modelling/$s.py
    done

Before the full preprocessing run, a quick smoke test takes a few minutes:

    sbatch euler/preprocess.sbatch --months 01 --limit-hours 48

The spatial subset has only about 200 hourly CLARA cells (none from May to July),
so `env.sh` sets smaller cross-validation folds through `ERA5_N_SPACE_FOLDS`,
`ERA5_N_TIME_FOLDS`, `ERA5_MIN_TRAIN_ROWS` and `ERA5_MIN_TEST_ROWS`. Each model log
prints `Skipped space fold …` for any fold still too small. Override them per job
with, for example, `ERA5_N_SPACE_FOLDS=4 sbatch euler/run_modelling.sbatch …`.

    bash euler/pull_results.sh                  # cluster -> laptop (results)

The fetch job skips months already downloaded, so if it hits the wall clock,
just `sbatch` it again and it resumes.

## Troubleshooting

### `Connection refused` to cds.climate.copernicus.eu

Euler compute nodes have no direct internet access. Without a proxy the fetch
job sits in a retry loop (`attempt N of 500`, 120 s apart) and downloads
nothing, while `myjobs` cheerfully reports RUNNING at ~0% CPU.

`env.sh` loads the `eth_proxy` module to fix this. If you see this error,
check that `eth_proxy` appears in `ERA5_MODULES` and that the job log shows no
warning about it failing to load. Login nodes DO have internet, which is why
the same request works when run by hand and fails under `sbatch`.

### Job is RUNNING but CPU utilization is ~0%

For the fetch job that is normal at first -- it waits on the CDS queue. But if
it stays at 0% with an empty `.out` log for a long time, read the `.err` file:
that is where the connection retries go.

### The `.out` log is empty while the job runs

Python buffers stdout when it is redirected to a file. Both sbatch scripts run
`python -u` to disable that, so progress lines appear live. If you write a new
job script, remember the `-u`.
