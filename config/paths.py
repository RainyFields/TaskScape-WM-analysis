"""Central paths for TaskScape-WM-analysis.

Inputs come only from the TaskScape-WM BIDS dataset; outputs go only to OUTPUT_ROOT. Nothing reads from
the frozen legacy project (`fmri_dataset_project`), and this module enforces that:

* ``DATASET_ROOT`` / ``OUTPUT_ROOT`` default to the canonical cluster locations and can be overridden with
  ``TASKSCAPE_DATASET_ROOT`` / ``TASKSCAPE_OUTPUT_ROOT`` (scripts may also pass ``--dataset_root`` /
  ``--output_root`` and call :func:`configure`).
* Any root or input path that resolves into the legacy tree raises ``SystemExit``.
* :func:`install_legacy_guard` adds a Python audit hook that fails any ``open()`` / ``os.listdir`` /
  ``os.scandir`` of a path inside the legacy tree and logs every opened path under ``/project/def-pbellec/xuan``
  to ``OUTPUT_ROOT/audit/<script>_<pid>.opened.tsv`` (evidence that no legacy input was used).

Dataset layout (built by TaskScape-WM-migration/scripts/build_taskscape_wm.py)::

    DATASET_ROOT/
      sub-XX/ses-YYY/func/*_events.tsv                      BIDS events (generated)
      sourcedata/behavior-reformatted/sub-XX/ses-YYY/func/  reformatted behaviour (GLM events input)
      sourcedata/behavior-corrected/  sourcedata/events-raw/  sourcedata/study-designs/
      derivatives/fmriprep/sub-XX/ses-YYY/func/*_desc-confounds_timeseries.tsv
      derivatives/taskscape-surface/sub-XX/ses-YYY/func/*_space-fsLR_den-32k[_desc-sm4]_bold.dtseries.nii
      derivatives/taskscape-surface/atlas/{Glasser_LR_Dense64k.dlabel.nii, valid_vertex_mask.npy, ...}
      derivatives/glm/sub-XX/ses-YYY/func/*_run-NN_{betas.h5, designmatrix.tsv, trials.tsv}
      derivatives/glm-clean-delay/sub-XX/sub-XX_{delay_betas.h5, provenance.csv, ...}
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

LEGACY_ROOTS = tuple(Path(p) for p in (
    "/project/def-pbellec/xuan/fmri_dataset_project",
    "/lustre09/project/6002071/xuan/fmri_dataset_project",
))
WATCH_PREFIXES = ("/project/def-pbellec/xuan", "/lustre09/project/6002071/xuan")

DEFAULT_DATASET_ROOT = Path("/project/def-pbellec/xuan/TaskScape-WM")
DEFAULT_OUTPUT_ROOT = Path("/project/def-pbellec/xuan/TaskScape-WM-analysis/outputs")
REPO_ROOT = Path(__file__).resolve().parents[1]

SURFACE_SPACE = "space-fsLR_den-32k"

# EXTERNAL (not part of TaskScape-WM, not legacy): stimulus/block-file repository used only to verify run labels
# (scripts/verify_run_labels_from_blockfiles.py). Recorded revision at migration: 4b88bce0dcf9 (2025-12-10).
TASK_STIMULI = Path(os.environ.get("TASKSCAPE_TASK_STIMULI", "/project/def-pbellec/xuan/task_stimuli/data/multfs"))
TASK_STIMULI_REVISION = "4b88bce0dcf90acc25d0feb2b89e0f6702d5f223"
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05", "sub-06")


def is_legacy(p: str | os.PathLike) -> bool:
    s = os.path.abspath(os.fspath(p))
    try:
        r = os.path.realpath(s)
    except OSError:
        r = s
    return any(x == str(L) or x.startswith(str(L) + os.sep) for x in (s, r) for L in LEGACY_ROOTS)


def require_not_legacy(p: str | os.PathLike, what: str = "path") -> Path:
    if is_legacy(p):
        raise SystemExit(f"[config.paths] {what} resolves into the frozen legacy project: {p}\n"
                         "TaskScape-WM-analysis reads inputs only from the TaskScape-WM dataset.")
    return Path(p)


class _Roots:
    dataset: Path
    output: Path


ROOTS = _Roots()


def configure(dataset_root: str | os.PathLike | None = None, output_root: str | os.PathLike | None = None,
              create_output: bool = True) -> _Roots:
    ds = Path(dataset_root or os.environ.get("TASKSCAPE_DATASET_ROOT") or DEFAULT_DATASET_ROOT)
    out = Path(output_root or os.environ.get("TASKSCAPE_OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT)
    require_not_legacy(ds, "DATASET_ROOT")
    require_not_legacy(out, "OUTPUT_ROOT")
    if not (ds / "dataset_description.json").is_file():
        raise SystemExit(f"[config.paths] DATASET_ROOT has no dataset_description.json: {ds}")
    if os.path.realpath(out).startswith(os.path.realpath(ds) + os.sep):
        raise SystemExit(f"[config.paths] OUTPUT_ROOT must not be inside the dataset: {out}")
    if create_output:
        out.mkdir(parents=True, exist_ok=True)
    ROOTS.dataset, ROOTS.output = ds, out
    return ROOTS


def _ds() -> Path:
    if not hasattr(ROOTS, "dataset"):
        configure()
    return ROOTS.dataset


def _out() -> Path:
    if not hasattr(ROOTS, "output"):
        configure()
    return ROOTS.output


def run_label(run: int | str) -> str:
    """'1', 1, 'run-1', 'run-01' -> 'run-01' (the dataset zero-pads every run label)."""
    m = re.search(r"(\d+)$", str(run))
    if not m:
        raise ValueError(f"bad run label {run!r}")
    return f"run-{int(m.group(1)):02d}"


# --- dataset inputs -----------------------------------------------------------------------------
def dataset_root() -> Path: return _ds()
def surface_dir() -> Path: return _ds() / "derivatives/taskscape-surface"
def atlas_dir() -> Path: return surface_dir() / "atlas"
def fmriprep_dir() -> Path: return _ds() / "derivatives/fmriprep"
def glm_dir() -> Path: return _ds() / "derivatives/glm"
def clean_delay_dir() -> Path: return _ds() / "derivatives/glm-clean-delay"
def reformatted_events_dir() -> Path: return _ds() / "sourcedata/behavior-reformatted"
def corrected_behavior_dir() -> Path: return _ds() / "sourcedata/behavior-corrected"
def study_designs_dir() -> Path: return _ds() / "sourcedata/study-designs"
def raw_events_dir() -> Path: return _ds() / "sourcedata/events-raw"
def unconverted_events_dir() -> Path: return _ds() / "sourcedata/events-unconverted"


def raw_events_files(sub: str) -> list[Path]:
    """All original raw events of a subject (verbatim): events-raw (converted runs) + events-unconverted
    (header-only exception runs). Together they equal the raw dataset's func/*_events.tsv minus __dup / pilot."""
    return sorted([*raw_events_dir().glob(f"{sub}/ses-*/func/*_events.tsv"),
                   *unconverted_events_dir().glob(f"{sub}/ses-*/func/*_events.tsv")], key=lambda p: p.name)
def valid_vertex_mask() -> Path: return atlas_dir() / "valid_vertex_mask.npy"
def glasser_dlabel() -> Path: return atlas_dir() / "Glasser_LR_Dense64k.dlabel.nii"
def parcel_networks() -> Path: return atlas_dir() / "cortex_parcel_network_assignments.txt"


def stem(sub: str, ses: str, task: str, run) -> str:
    return f"{sub}_{ses}_task-{task}_{run_label(run)}"


def surface_bold(sub: str, ses: str, task: str, run, smoothed: bool = True) -> Path:
    desc = "_desc-sm4" if smoothed else ""
    return surface_dir() / sub / ses / "func" / f"{stem(sub, ses, task, run)}_{SURFACE_SPACE}{desc}_bold.dtseries.nii"


def confounds(sub: str, ses: str, task: str, run) -> Path:
    return fmriprep_dir() / sub / ses / "func" / f"{stem(sub, ses, task, run)}_desc-confounds_timeseries.tsv"


def reformatted_events(sub: str, ses: str, task: str, run) -> Path:
    return reformatted_events_dir() / sub / ses / "func" / f"{stem(sub, ses, task, run)}_events.tsv"


def glm_files(sub: str, ses: str, task: str, run) -> dict[str, Path]:
    d = glm_dir() / sub / ses / "func"
    s = stem(sub, ses, task, run)
    return {"betas": d / f"{s}_betas.h5", "designmatrix": d / f"{s}_designmatrix.tsv", "trials": d / f"{s}_trials.tsv"}


# --- outputs ------------------------------------------------------------------------------------
def output_root() -> Path: return _out()


def output_dir(*parts: str) -> Path:
    p = _out().joinpath(*parts)
    require_not_legacy(p, "output path")
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- runtime guard against legacy fallback ------------------------------------------------------
_AUDIT = {"installed": False, "fh": None}
_CODE_SUFFIXES = (".py", ".pyc", ".so", ".pth", ".mplstyle", ".ttf", ".otf")


def install_legacy_guard(script: str | None = None) -> Path:
    """Fail any open()/listdir/scandir of a legacy-tree path; log every opened path under WATCH_PREFIXES."""
    if _AUDIT["installed"]:
        return Path(_AUDIT["fh"].name)
    log = output_dir("audit") / f"{Path(script or sys.argv[0]).stem}_{os.getpid()}.opened.tsv"
    fh = open(log, "w", buffering=1)
    fh.write("event\tpath\tmode\n")
    _AUDIT.update(installed=True, fh=fh)

    def hook(event, args):
        if event not in ("open", "os.listdir", "os.scandir"):
            return
        path = args[0] if args else None
        if isinstance(path, int) or path is None:
            return
        try:
            s = os.fsdecode(path)
        except TypeError:
            return
        if is_legacy(s):
            if not fh.closed:
                fh.write(f"BLOCKED\t{s}\t\n")
            raise PermissionError(f"[config.paths] legacy-tree access blocked: {s}")
        if s.startswith(WATCH_PREFIXES) and not s.endswith(_CODE_SUFFIXES) and "/nilearn/lib/" not in s:
            if fh.closed:
                return
            mode = args[1] if event == "open" and len(args) > 1 else ""
            fh.write(f"{event}\t{s}\t{mode}\n")

    sys.addaudithook(hook)
    return log
