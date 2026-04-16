#!/usr/bin/env python3
"""
PureProtX Blockchain Overhead Baseline

Measures the screening pipeline with and without blockchain commits
to isolate the exact overhead of PureChain anchoring.

Mode A: pipeline only (blockchain disabled)
Mode B: pipeline + PureChain commit (full stack)

Reports median and P95 across 5 repeats for each target.

Output: results/overhead_baseline.json
"""

import os
import sys
import json
import time
import hashlib
import logging

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def measure_pipeline(target: str, n_compounds: int = 1000,
                     n_repeats: int = 5, use_blockchain: bool = True) -> list:
    """Run the screening pipeline n_repeats times, return list of wall-clock seconds."""
    import joblib

    model_path = os.path.join(
        PROJECT_ROOT, "experiments", "paper_results", "models",
        f"{target}_model.joblib",
    )
    model_data = joblib.load(model_path)
    scaler = model_data["scaler"]
    models = model_data["models"]

    connector = None
    if use_blockchain:
        from blockchain.purechain_factory import get_purechain_connector
        connector = get_purechain_connector(strict=False)

    rng = np.random.RandomState(42)

    times = []
    for rep in range(n_repeats):
        X = rng.randn(n_compounds, 2058).astype(np.float32)

        t0 = time.perf_counter()

        # Featurize (scaler transform)
        X_scaled = scaler.transform(X)

        # Consensus inference
        preds = np.mean([m.predict(X_scaled) for m in models.values()], axis=0)

        # Canonical JSON + SHA-256
        canonical = json.dumps(
            {"target": target, "n": n_compounds,
             "predictions": [round(float(p), 4) for p in preds]},
            sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()

        # Blockchain commit (if enabled)
        if use_blockchain and connector:
            result_hash = bytes.fromhex(digest)
            data_hash = hashlib.sha256(f"overhead_{target}_{rep}".encode()).digest()
            connector.record_and_verify_result(
                result_hash, data_hash, f"overhead_{target}_{rep}"
            )

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mode = "BC" if use_blockchain else "NO-BC"
        print(f"    [{mode}] rep {rep+1}/{n_repeats}: {elapsed:.3f} s")

    return times


def run_overhead_baseline(targets=None, n_compounds: int = 1000,
                          n_repeats: int = 5) -> dict:
    """Run the full overhead baseline for given targets."""
    if targets is None:
        targets = ["CHEMBL243", "CHEMBL240"]

    results = {}

    for target in targets:
        print(f"\n  [{target}] Pipeline only (no blockchain)...")
        times_no_bc = measure_pipeline(target, n_compounds, n_repeats, use_blockchain=False)

        print(f"  [{target}] Pipeline + PureChain commit...")
        times_with_bc = measure_pipeline(target, n_compounds, n_repeats, use_blockchain=True)

        a_no = np.array(times_no_bc)
        a_bc = np.array(times_with_bc)
        overhead_s = float(np.median(a_bc) - np.median(a_no))
        overhead_pct = (overhead_s / np.median(a_no)) * 100 if np.median(a_no) > 0 else 0

        results[target] = {
            "target": target,
            "n_compounds": n_compounds,
            "n_repeats": n_repeats,
            "pipeline_only_median_s": round(float(np.median(a_no)), 4),
            "pipeline_only_p95_s": round(float(np.percentile(a_no, 95)), 4),
            "pipeline_with_blockchain_median_s": round(float(np.median(a_bc)), 4),
            "pipeline_with_blockchain_p95_s": round(float(np.percentile(a_bc, 95)), 4),
            "blockchain_overhead_s": round(overhead_s, 4),
            "blockchain_overhead_pct": round(overhead_pct, 1),
        }

        print(f"    Pipeline only:   {np.median(a_no):.3f} s (median)")
        print(f"    With blockchain: {np.median(a_bc):.3f} s (median)")
        print(f"    Overhead:        {overhead_s:.3f} s ({overhead_pct:.1f}%)")

    # Save
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "overhead_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX Blockchain Overhead Baseline")
    print("=" * 60)
    run_overhead_baseline()
