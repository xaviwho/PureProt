#!/usr/bin/env python3
"""
A1 orchestrator — cross-ORT-version determinism matrix (Reviewer R2.4).

For each ONNX Runtime version in the sweep, creates an isolated venv, installs
`onnxruntime==<ver>` (+ numpy), runs experiments/determinism_harness.py, and
collects the per-version JSON. Then aggregates all versions into a hash-agreement
matrix. Install failures (e.g. no cp312 wheel for old ORT) are recorded as
`unavailable`, not hidden.

Runs on x86 (Windows dev box). The SAME harness runs on the Jetson in B1 to add
the aarch64 column.

Outputs (results/determinism/):
  harness_ort<ver>.json         per-version raw harness output
  determinism_matrix.md / .csv  aggregated cross-version / cross-thread matrix
  matrix_run.log                (this script's stdout, if teed by caller)

Run with system Python 3.12.10:
  python experiments/run_determinism_matrix.py
"""

import json
import os
import subprocess
import sys

ORT_VERSIONS = ["1.16.3", "1.17.3", "1.18.0", "1.19.2", "1.20.1", "1.22.0"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_BASE = os.path.join(ROOT, "experiments", ".det_venvs")
OUT_DIR = os.path.join(ROOT, "results", "determinism")
HARNESS = os.path.join(ROOT, "experiments", "determinism_harness.py")


def venv_python(vdir):
    return (os.path.join(vdir, "Scripts", "python.exe") if os.name == "nt"
            else os.path.join(vdir, "bin", "python"))


def build_and_run(ver):
    tag = ver.replace(".", "_")
    vdir = os.path.join(VENV_BASE, f"ort_{tag}")
    vpy = venv_python(vdir)
    out_json = os.path.join(OUT_DIR, f"harness_ort{ver}.json")
    rec = {"ort_version": ver, "status": None, "detail": "", "json": out_json}

    if not os.path.exists(vpy):
        r = subprocess.run([sys.executable, "-m", "venv", vdir],
                           capture_output=True, text=True)
        if r.returncode != 0:
            rec["status"] = "venv_failed"; rec["detail"] = r.stderr[-400:]; return rec

    # Install exactly the pinned ORT version (+ numpy, resolved & recorded).
    r = subprocess.run(
        [vpy, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
         f"onnxruntime=={ver}", "numpy"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        rec["status"] = "install_failed"
        # keep the salient pip line (usually 'Could not find a version…')
        tail = (r.stderr or r.stdout).strip().splitlines()
        rec["detail"] = " | ".join(tail[-3:])[:500]
        print(f"[{ver}] install FAILED: {rec['detail']}")
        return rec

    # ABI check: older ORT builds are compiled against numpy 1.x and crash under
    # numpy 2.x. If import fails, pin numpy<2 and retry (recorded, not hidden).
    t = subprocess.run([vpy, "-c", "import onnxruntime"], capture_output=True, text=True)
    if t.returncode != 0:
        subprocess.run([vpy, "-m", "pip", "install", "--quiet", "numpy<2"],
                       capture_output=True, text=True)
        t = subprocess.run([vpy, "-c", "import onnxruntime"], capture_output=True, text=True)
        rec["numpy_pinned_lt2"] = True
        if t.returncode != 0:
            rec["status"] = "import_failed"
            rec["detail"] = (t.stderr or t.stdout)[-500:]
            print(f"[{ver}] import FAILED after numpy<2: {rec['detail'][:120]}")
            return rec

    r = subprocess.run(
        [vpy, HARNESS, "--onnx-dir", "models/onnx", "--out", out_json],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        rec["status"] = "run_failed"; rec["detail"] = (r.stderr or r.stdout)[-500:]
        print(f"[{ver}] harness FAILED: {rec['detail']}")
        return rec

    rec["status"] = "ok"
    print(f"[{ver}] OK")
    print(r.stdout.strip())
    return rec


def aggregate(records):
    loaded = {}
    for rec in records:
        if rec["status"] == "ok" and os.path.exists(rec["json"]):
            with open(rec["json"]) as f:
                loaded[rec["ort_version"]] = json.load(f)

    # model list from any successful run
    models = sorted({m for d in loaded.values() for m in d["results"]})
    versions = [v for v in ORT_VERSIONS if v in loaded]

    lines = ["# A1 — ONNX Determinism Matrix (R2.2, R2.4)\n"]
    lines.append("Cross-ORT-version × thread × 40-run bitwise-hash agreement on "
                 "the exported models. Same harness runs on aarch64 (Jetson) in B1.\n")

    # environment table
    lines.append("## Environments\n")
    lines.append("| ORT | status | numpy | python | platform |")
    lines.append("|---|---|---|---|---|")
    for rec in records:
        v = rec["ort_version"]
        if v in loaded:
            m = loaded[v]["manifest"]
            lines.append(f"| {v} | ok | {m['numpy_version']} | {m['python_version']} | {m['machine']}/{m['system']} |")
        else:
            lines.append(f"| {v} | **{rec['status']}** | — | — | {rec['detail'][:60]} |")
    lines.append("")

    # within-config + cross-thread (per version)
    lines.append("## Within-config determinism & cross-thread agreement\n")
    lines.append("Each cell: are all 40 runs identical (within-config), and does the "
                 "hash stay equal across threads {1,2,4} (cross-thread)?\n")
    lines.append("| Model | " + " | ".join(versions) + " |")
    lines.append("|---" * (len(versions) + 1) + "|")
    for model in models:
        row = [model]
        for v in versions:
            cells = loaded[v]["results"].get(model, {})
            within = all(c["bitwise_identical"] for c in cells.values()) if cells else None
            hashes = {c["canonical_hash"] for c in cells.values()}
            cross_thread = len(hashes) == 1
            if within is None:
                row.append("—")
            else:
                row.append(("det" if within else "NON-DET") + ("/thr-stable" if cross_thread else "/THR-VARY"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # cross-ORT agreement (threads=1 canonical hash)
    lines.append("## Cross-ORT-version agreement (threads=1)\n")
    lines.append("Distinct canonical hashes for each model across the successful ORT "
                 "versions. **1 = byte-identical across every ORT version.**\n")
    lines.append("| Model | distinct hashes across ORT | verdict |")
    lines.append("|---|---|---|")
    for model in models:
        hs = set()
        for v in versions:
            c = loaded[v]["results"].get(model, {}).get("threads_1")
            if c:
                hs.add(c["canonical_hash"])
        verdict = "stable across ORT" if len(hs) == 1 else f"**DRIFTS ({len(hs)} variants)**"
        lines.append(f"| {model} | {len(hs)} | {verdict} |")
    lines.append("")
    lines.append("_Note: numpy co-varies with ORT where old ORT pins numpy<2; see "
                 "Environments table. Hash drift may reflect ORT and/or numpy changes._")

    with open(os.path.join(OUT_DIR, "determinism_matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV: model, ort, threads, bitwise_identical, canonical_hash, latency
    import csv
    with open(os.path.join(OUT_DIR, "determinism_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ort_version", "numpy", "threads", "bitwise_identical",
                    "canonical_hash", "latency_ms_mean"])
        for v in versions:
            npv = loaded[v]["manifest"]["numpy_version"]
            for model, cells in loaded[v]["results"].items():
                for tk, c in cells.items():
                    w.writerow([model, v, npv, c["threads"], c["bitwise_identical"],
                                c["canonical_hash"], c["latency_ms_mean"]])
    print(f"[aggregate] wrote determinism_matrix.md + .csv ({len(versions)} ORT versions)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VENV_BASE, exist_ok=True)
    print(f"[matrix] base python: {sys.executable} ({sys.version.split()[0]})")
    records = []
    for ver in ORT_VERSIONS:
        print(f"\n===== ORT {ver} =====")
        records.append(build_and_run(ver))
    with open(os.path.join(OUT_DIR, "matrix_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    aggregate(records)
    ok = [r["ort_version"] for r in records if r["status"] == "ok"]
    print(f"\n[matrix] done. successful ORT versions: {ok}")


if __name__ == "__main__":
    main()
