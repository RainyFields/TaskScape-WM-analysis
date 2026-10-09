#!/usr/bin/env python3
"""Read-only access check for the shared TaskScape-WM dataset on Rorqual. Run it under YOUR account.

    python3 scripts/check_dataset_access.py                 # representative files (seconds)
    python3 scripts/check_dataset_access.py --full          # + os.access(R_OK) on every file and symlink target

Uses only the Python standard library (any python3 works); if nibabel / h5py are importable it also reads a header /
a small slice through them. Reads at most a few kB per file; never writes. Exit code 0 = every check passed.

What it covers (one representative file per kind, all for sub-01/ses-001 ctxdm run-01):
  traversal of every path component; participants.tsv; BIDS events; raw BOLD NIfTI (symlink into the CNeuroMod RIA
  store, needs group rrg-pbellec); fMRIPrep preprocessed BOLD (RIA symlink) + confounds; surface CIFTI (hardlink);
  complete GLM betas H5 + design matrix + trial table; clean-delay betas H5.
"""
from __future__ import annotations

import argparse
import grp
import gzip
import os
import pwd
import struct
import sys
from pathlib import Path

DEFAULT_ROOT = os.environ.get("TASKSCAPE_DATASET_ROOT", "/project/def-pbellec/xuan/TaskScape-WM")
STEM = "sub-01_ses-001_task-ctxdm_run-01"
FILES = {
    "participants": "participants.tsv",
    "events (BIDS)": f"sub-01/ses-001/func/{STEM}_events.tsv",
    "raw BOLD NIfTI (RIA symlink)": f"sub-01/ses-001/func/{STEM}_bold.nii.gz",
    "fMRIPrep preproc BOLD (RIA symlink)": f"derivatives/fmriprep/sub-01/ses-001/func/{STEM}_space-T1w_desc-preproc_bold.nii.gz",
    "fMRIPrep confounds": f"derivatives/fmriprep/sub-01/ses-001/func/{STEM}_desc-confounds_timeseries.tsv",
    "surface CIFTI sm4 (hardlink)": f"derivatives/taskscape-surface/sub-01/ses-001/func/{STEM}_space-fsLR_den-32k_desc-sm4_bold.dtseries.nii",
    "GLM betas H5 (hardlink)": f"derivatives/glm/sub-01/ses-001/func/{STEM}_betas.h5",
    "GLM design matrix": f"derivatives/glm/sub-01/ses-001/func/{STEM}_designmatrix.tsv",
    "GLM trial table": f"derivatives/glm/sub-01/ses-001/func/{STEM}_trials.tsv",
    "clean-delay betas H5 (hardlink)": "derivatives/glm-clean-delay/sub-01/sub-01_delay_betas.h5",
}
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        results.append((name, True, detail))
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        results.append((name, False, f"{type(e).__name__}: {e}"))


def tsv_info(p: Path) -> str:
    with open(p) as f:
        header = f.readline().rstrip("\n").split("\t")
        n = sum(1 for _ in f)
    return f"{n} rows, {len(header)} columns ({', '.join(header[:4])}, ...)"


def nifti1_gz(p: Path) -> str:
    with gzip.open(p, "rb") as f:
        h = f.read(348)
    if struct.unpack("<i", h[:4])[0] != 348:
        raise ValueError("not a NIfTI-1 header")
    dims = struct.unpack("<8h", h[40:56])
    extra = ""
    try:
        import nibabel as nib
        extra = f"; nibabel shape {nib.load(str(p)).shape}"
    except ImportError:
        pass
    return f"NIfTI-1 dims {dims[1:dims[0] + 1]}{extra}"


def cifti(p: Path) -> str:
    with open(p, "rb") as f:
        h = f.read(540)
    if struct.unpack("<i", h[:4])[0] != 540:
        raise ValueError("not a NIfTI-2/CIFTI header")
    dims = struct.unpack("<8q", h[16:80])
    return f"NIfTI-2 dims {dims[1:dims[0] + 1]}"


def hdf5(p: Path) -> str:
    with open(p, "rb") as f:
        if f.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError("no HDF5 signature")
    try:
        import h5py
    except ImportError:
        return "HDF5 signature OK (h5py not importable: slice not read)"
    with h5py.File(p, "r") as f:
        b = f["betas"]
        sl = b[:3, :3]
        k = dict(f.attrs).get("K_task", "n/a")
        return f"betas {b.shape} {b.dtype}, K_task={k}, slice[:3,:3] read ({sl.size} values)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--full", action="store_true", help="also test read permission of every file / symlink target")
    a = ap.parse_args()
    root = Path(a.root)
    user = pwd.getpwuid(os.getuid()).pw_name
    groups = sorted(grp.getgrgid(g).gr_name for g in os.getgroups())
    print(f"user={user} groups={','.join(groups)}\nroot={root} (realpath {os.path.realpath(root)})")
    for need in ("def-pbellec", "rrg-pbellec"):
        results.append((f"member of {need}", need in groups,
                        "needed for the dataset" if need == "def-pbellec" else "needed for raw/fMRIPrep payloads (RIA store)"))

    def traverse():
        parts = root.parts
        for i in range(2, len(parts) + 1):
            d = Path(*parts[:i])
            if not os.access(d, os.X_OK):
                raise PermissionError(f"cannot traverse {d}")
        return f"{len(os.listdir(root))} entries listed at the root"
    check("traverse + list dataset root", traverse)

    readers = {"participants": tsv_info, "events (BIDS)": tsv_info, "fMRIPrep confounds": tsv_info,
               "GLM design matrix": tsv_info, "GLM trial table": tsv_info,
               "raw BOLD NIfTI (RIA symlink)": nifti1_gz, "fMRIPrep preproc BOLD (RIA symlink)": nifti1_gz,
               "surface CIFTI sm4 (hardlink)": cifti, "GLM betas H5 (hardlink)": hdf5,
               "clean-delay betas H5 (hardlink)": hdf5}
    for name, rel in FILES.items():
        p = root / rel
        def fn(p=p, name=name):
            tgt = f" -> {os.path.realpath(p)}" if p.is_symlink() else ""
            return readers[name](p) + tgt[:90]
        check(name, fn)

    if a.full:
        def full():
            bad, n = [], 0
            for dp, dns, fns in os.walk(root, followlinks=False):
                if not os.access(dp, os.R_OK | os.X_OK):
                    bad.append(dp)
                for f in fns:
                    n += 1
                    if not os.access(os.path.join(dp, f), os.R_OK):  # follows symlinks to the RIA object
                        bad.append(os.path.join(dp, f))
            if bad:
                raise PermissionError(f"{len(bad)} of {n} not readable, e.g. {bad[:3]}")
            return f"all {n} files (incl. symlink targets) and all directories readable"
        check("full read-permission scan", full)

    w = max(len(r[0]) for r in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
    n_fail = sum(not r[1] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
