"""Central resolution of data and results locations.

Both roots can be overridden with environment variables so the same code runs
unchanged on a laptop and on a cluster, where the large files must live on a
scratch/work filesystem rather than next to the source:

    export ERA5_DATA_ROOT=/cluster/work/<group>/<user>/era5/data
    export ERA5_RESULTS_ROOT=/cluster/work/<group>/<user>/era5/results

With neither set, both fall back to the repository checkout, which reproduces
the current local layout.

Each data subset (``spatial_subset``: Italy, all of 2020; ``temporal_subset``:
global, 2020-12-01..05) keeps its own files under ``<root>/<subset>/``. The
active one is chosen with

    export ERA5_SUBSET=temporal_subset     # default: spatial_subset

and reached through ``subset_data_path`` / ``subset_results_path``. Files
shared by every subset (``CLARA.pkl``) stay on ``data_path``.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _root(env_var: str, default_name: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else REPO_ROOT / default_name


DATA_ROOT = _root("ERA5_DATA_ROOT", "data")
RESULTS_ROOT = _root("ERA5_RESULTS_ROOT", "results")

SUBSETS = ("spatial_subset", "temporal_subset")
SUBSET = os.environ.get("ERA5_SUBSET", "spatial_subset")
if SUBSET not in SUBSETS:
    raise ValueError(f"ERA5_SUBSET must be one of {SUBSETS}, got {SUBSET!r}")


def data_path(*parts: str) -> Path:
    """Absolute path to a file under the data root."""
    return DATA_ROOT.joinpath(*parts)


def results_path(*parts: str) -> Path:
    """Absolute path under the results root, creating the parent directory."""
    path = RESULTS_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def subset_data_path(*parts: str, subset: str = None) -> Path:
    """Absolute path under the data root of the active (or given) subset."""
    return data_path(subset or SUBSET, *parts)


def subset_results_path(*parts: str, subset: str = None) -> Path:
    """Absolute path under the results root of the active (or given) subset,
    creating the parent directory."""
    return results_path(subset or SUBSET, *parts)


def show_or_save(plt, name: str, dpi: int = 150) -> Path:
    """Save the current figure under the results root, and display it too when
    a screen is available.

    On a cluster node there is no display, so a bare ``plt.show()`` throws the
    figure away silently. Calling this instead means the plot always survives
    as a file, and still pops up when you run the script on your laptop.
    """
    out = name if isinstance(name, Path) else results_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    import matplotlib
    interactive = matplotlib.get_backend().lower() not in {
        "agg", "pdf", "svg", "ps", "cairo", "template"
    }
    if interactive and (os.environ.get("DISPLAY") or sys.platform == "darwin"):
        plt.show()
    plt.close()
    print(f"figure -> {out}")
    return out
