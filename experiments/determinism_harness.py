#!/usr/bin/env python3
"""
A1 — ONNX Determinism Characterization Harness (Reviewer R2.2, R2.4)

Cross-platform, self-contained. Runs the exported ONNX models under a sweep of
thread counts, N times each, and records the SHA-256 of every output so that
bitwise reproducibility can be checked:

  * within-config  — are all N runs of one (model, threads) cell identical?
  * cross-thread    — does a model hash the same at 1 vs 2 vs 4 threads?
  * cross-runtime   — (aggregated across ORT-version venvs by the orchestrator)
  * cross-arch      — (aggregated later: the SAME script runs on the Jetson, B1)

Deliberately depends on ONLY onnxruntime + numpy so it installs identically
under every ORT version venv (x86) and on aarch64 (Jetson). The fixed seeded
input matches export/verify_onnx.py (RandomState(42), 2058 features) so hashes
are comparable to the V-F baseline.

Output: one JSON per environment (ORT version tagged), with a full manifest.

Run:
  python experiments/determinism_harness.py --onnx-dir models/onnx --out results/determinism/harness.json
"""

import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np

N_FEATURES = 2058          # matches export/verify_onnx.py
SEED = 42
DEFAULT_THREADS = [1, 2, 4]


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def make_input(n_samples: int) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.randn(n_samples, N_FEATURES)


def run_cell(onnx_path, X, threads, runs):
    """Run one (model, thread-count) cell `runs` times; return hash stats."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    # Force a stable, single execution provider.
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    info = sess.get_inputs()[0]
    name = info.name
    X_in = X.astype(np.float64) if info.type == "tensor(double)" else X.astype(np.float32)

    hashes = []
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = sess.run(None, {name: X_in})[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        hashes.append(sha256_array(out))

    uniq = sorted(set(hashes))
    return {
        "threads": threads,
        "input_dtype": "float64" if info.type == "tensor(double)" else "float32",
        "output_shape": list(np.asarray(sess.run(None, {name: X_in})[0]).shape),
        "runs": runs,
        "unique_hashes": len(uniq),
        "bitwise_identical": len(uniq) == 1,
        "canonical_hash": uniq[0],           # first (sorted) hash of the cell
        "all_hashes_sample": uniq[:3],       # >1 only if non-deterministic
        "latency_ms_mean": round(float(np.mean(latencies)), 3),
    }


def manifest(onnx_dir, n_samples, runs, threads):
    import onnxruntime as ort
    return {
        "experiment": "A1 determinism harness",
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "onnx_dir": onnx_dir,
        "n_samples": n_samples,
        "runs_per_cell": runs,
        "thread_sweep": threads,
        "input_seed": SEED,
        "n_features": N_FEATURES,
        "available_providers": ort.get_available_providers(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models/onnx")
    ap.add_argument("--out", default="results/determinism/harness.json")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--threads", default=",".join(map(str, DEFAULT_THREADS)))
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_dir = args.onnx_dir if os.path.isabs(args.onnx_dir) else os.path.join(root, args.onnx_dir)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    threads = [int(t) for t in args.threads.split(",")]

    X = make_input(args.n_samples)
    files = sorted(f for f in os.listdir(onnx_dir) if f.endswith(".onnx"))
    if not files:
        raise SystemExit(f"no .onnx files in {onnx_dir}")

    mani = manifest(args.onnx_dir, args.n_samples, args.runs, threads)
    print(f"[harness] ORT {mani['onnxruntime_version']} numpy {mani['numpy_version']} "
          f"{mani['machine']} py{mani['python_version']}")

    results = {}
    for fn in files:
        model = os.path.splitext(fn)[0]
        path = os.path.join(onnx_dir, fn)
        results[model] = {}
        for t in threads:
            cell = run_cell(path, X, t, args.runs)
            results[model][f"threads_{t}"] = cell
            flag = "OK " if cell["bitwise_identical"] else "NON-DET"
            print(f"  {model:24} t={t}: {flag} hash={cell['canonical_hash'][:16]} "
                  f"({cell['latency_ms_mean']:.1f} ms)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"manifest": mani, "results": results}, f, indent=2)
    print(f"[harness] wrote {out_path}")


if __name__ == "__main__":
    main()
