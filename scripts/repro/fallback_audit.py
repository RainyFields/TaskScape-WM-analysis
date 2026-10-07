#!/usr/bin/env python3
"""Fallback audit for a fidelity run: every path opened by every job (strace open/openat, which also sees
HDF5/C-library opens that Python audit hooks miss) and every config.paths audit log. Pass = zero opens of
any path under the frozen legacy project (attempted or successful) and zero BLOCKED events; positive control =
opens under the TaskScape-WM dataset. Writes <run>/compare/fallback_audit.{json,tsv}."""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

LEGACY = ("/project/def-pbellec/xuan/fmri_dataset_project", "/lustre09/project/6002071/xuan/fmri_dataset_project")
DATASET = ("/project/def-pbellec/xuan/TaskScape-WM/", "/lustre09/project/6002071/xuan/TaskScape-WM/")
OPEN_RE = re.compile(r'open(?:at2?)?\((?:[A-Z_]+|\d+), "([^"]+)"|open\("([^"]+)"')

ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); a = ap.parse_args()
rows, tot = [], Counter()
for tr in sorted((a.run / "strace").glob("*.trace")):
    leg, ds, n = Counter(), 0, 0
    for line in tr.read_text(errors="replace").splitlines():
        m = OPEN_RE.search(line)
        if not m:
            continue
        p = m.group(1) or m.group(2); n += 1
        if p.startswith(LEGACY):
            leg[p] += 1
        elif p.startswith(DATASET):
            ds += 1
    rows.append({"job": tr.stem, "opens": n, "dataset_opens": ds, "legacy_opens": sum(leg.values()), "legacy_examples": list(leg)[:5]})
    tot["legacy"] += sum(leg.values()); tot["dataset"] += ds
blocked = []
for f in sorted((a.run / "out" / "audit").glob("*.opened.tsv")):
    blocked += [f"{f.name}: {l}" for l in f.read_text().splitlines() if l.startswith("BLOCKED")]
summ = {"n_traces": len(rows), "legacy_opens_total": tot["legacy"], "dataset_opens_total": tot["dataset"],
        "n_blocked_events": len(blocked), "blocked_examples": blocked[:10],
        "jobs_with_legacy_opens": [r["job"] for r in rows if r["legacy_opens"]],
        "jobs_without_dataset_opens": [r["job"] for r in rows if not r["dataset_opens"]],
        "PASS": tot["legacy"] == 0 and not blocked and len(rows) > 0}
(a.run / "compare").mkdir(exist_ok=True)
(a.run / "compare" / "fallback_audit.json").write_text(json.dumps(summ, indent=2))
with open(a.run / "compare" / "fallback_audit.tsv", "w") as fh:
    fh.write("job\topens\tdataset_opens\tlegacy_opens\tlegacy_examples\n")
    for r in rows:
        fh.write(f"{r['job']}\t{r['opens']}\t{r['dataset_opens']}\t{r['legacy_opens']}\t{r['legacy_examples']}\n")
print(json.dumps(summ, indent=2))
