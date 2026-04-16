#!/usr/bin/env python3
"""
PureProtX Blockchain Scalability Benchmark

Measures per-record blockchain anchoring latency as a function of batch size N,
comparing two strategies:
  Strategy A: N individual transactions (one per compound)
  Strategy B: 1 Merkle batch transaction (MerkleRoot of N hashes)

Identifies the crossover point where batching becomes advantageous -- a key
result for IoT deployments with high-frequency screening.

Output:
  results/scalability_benchmark.csv
  results/scalability_figure.png
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Merkle Tree
# ------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    """Return raw SHA-256 digest."""
    return hashlib.sha256(data).digest()


def compute_merkle_root(hashes: List[bytes]) -> bytes:
    """
    Compute the Merkle root from a list of 32-byte SHA-256 digests.

    Duplicates the last element when the layer has an odd count (standard
    Bitcoin-style Merkle).
    """
    if not hashes:
        return _sha256(b"")
    layer = list(hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            next_layer.append(_sha256(layer[i] + layer[i + 1]))
        layer = next_layer
    return layer[0]


# ------------------------------------------------------------------
# Synthetic hash generation
# ------------------------------------------------------------------

def _generate_compound_hashes(n: int, seed: int = 42) -> List[bytes]:
    """Generate N deterministic synthetic compound hashes."""
    rng = np.random.RandomState(seed)
    hashes = []
    for i in range(n):
        payload = f"compound_{i}_{rng.randint(0, 2**31)}".encode()
        hashes.append(_sha256(payload))
    return hashes


# ------------------------------------------------------------------
# Blockchain helpers (with graceful offline fallback)
# ------------------------------------------------------------------

def _get_connector():
    """Try to create a PurechainConnector; return None if offline."""
    from blockchain.purechain_factory import get_purechain_connector
    return get_purechain_connector(strict=False)


def _commit_hash(connector, result_hash: bytes, molecule_id: str) -> Dict[str, Any]:
    """Commit a single hash on-chain. Returns result dict with timing."""
    data_hash = _sha256(molecule_id.encode())
    t0 = time.perf_counter()
    result = connector.record_and_verify_result(result_hash, data_hash, molecule_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result["latency_ms"] = elapsed_ms
    return result


def _commit_hash_offline(result_hash: bytes, molecule_id: str) -> Dict[str, Any]:
    """Simulate a blockchain commit when no chain is available."""
    data_hash = _sha256(molecule_id.encode())
    t0 = time.perf_counter()
    # Simulate signing + serialisation + write overhead
    _ = hashlib.sha256(result_hash + data_hash + molecule_id.encode()).hexdigest()
    # Simulate network round-trip (local Ganache ~2-8ms, PureChain ~15-50ms)
    time.sleep(np.random.uniform(0.002, 0.010))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "success": True,
        "tx_hash": hashlib.sha256(result_hash).hexdigest()[:16],
        "latency_ms": elapsed_ms,
        "offline": True,
    }


# ------------------------------------------------------------------
# Benchmark core
# ------------------------------------------------------------------

def benchmark_anchoring_scalability(
    batch_sizes: List[int] = None,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    For each N in batch_sizes:
      - Strategy A: submit N individual hash commits, record total time
      - Strategy B: compute MerkleRoot of N hashes, submit 1 commit

    Returns DataFrame with columns:
      N, strategy_a_total_s, strategy_b_total_s,
      strategy_a_per_record_ms, strategy_b_per_record_ms,
      speedup_factor, crossover (bool: True where B first beats A)

    Saves to results/scalability_benchmark.csv
    Also saves a matplotlib figure to results/scalability_figure.png
    """
    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]

    connector = _get_connector()
    offline = connector is None
    if offline:
        print("  [scalability] Running in OFFLINE simulation mode")
    else:
        print("  [scalability] Connected to blockchain")

    rows = []

    for N in batch_sizes:
        print(f"  N={N:>6d} ...", end="", flush=True)
        hashes = _generate_compound_hashes(N)

        a_totals = []
        b_totals = []

        for rep in range(n_repeats):
            # --- Strategy A: N individual commits ---
            t0 = time.perf_counter()
            for i, h in enumerate(hashes):
                mol_id = f"bench_A_{N}_{rep}_{i}"
                if connector:
                    _commit_hash(connector, h, mol_id)
                else:
                    _commit_hash_offline(h, mol_id)
            a_total = time.perf_counter() - t0
            a_totals.append(a_total)

            # --- Strategy B: 1 Merkle commit ---
            t0 = time.perf_counter()
            merkle_root = compute_merkle_root(hashes)
            mol_id = f"bench_B_{N}_{rep}_merkle"
            if connector:
                _commit_hash(connector, merkle_root, mol_id)
            else:
                _commit_hash_offline(merkle_root, mol_id)
            b_total = time.perf_counter() - t0
            b_totals.append(b_total)

        a_mean = float(np.mean(a_totals))
        b_mean = float(np.mean(b_totals))
        a_per_rec = (a_mean / N) * 1000  # ms
        b_per_rec = (b_mean / N) * 1000  # ms
        speedup = a_mean / b_mean if b_mean > 0 else float("inf")

        rows.append({
            "N": N,
            "strategy_a_total_s": round(a_mean, 4),
            "strategy_b_total_s": round(b_mean, 4),
            "strategy_a_per_record_ms": round(a_per_rec, 4),
            "strategy_b_per_record_ms": round(b_per_rec, 4),
            "speedup_factor": round(speedup, 2),
            "crossover": False,  # filled below
        })
        print(f"  A={a_mean:.3f}s  B={b_mean:.3f}s  speedup={speedup:.1f}x")

    df = pd.DataFrame(rows)

    # Mark the crossover point (first N where Strategy B beats A)
    crossover_found = False
    for i, row in df.iterrows():
        if not crossover_found and row["strategy_b_total_s"] < row["strategy_a_total_s"]:
            df.at[i, "crossover"] = True
            crossover_found = True

    # --- Save CSV ---
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "scalability_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV -> {csv_path}")

    # --- Print crossover ---
    crossover_rows = df[df["crossover"]]
    if not crossover_rows.empty:
        cn = int(crossover_rows.iloc[0]["N"])
        print(f"  Crossover N = {cn} (Merkle batching first beats individual commits)")
    else:
        print("  Merkle batching is advantageous at all tested batch sizes")

    # --- Generate figure ---
    _plot_scalability(df, results_dir, offline)

    return df


def _plot_scalability(df: pd.DataFrame, results_dir: str, offline: bool) -> None:
    """Plot per-record latency vs N with crossover marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df["N"], df["strategy_a_per_record_ms"],
        "o-", color="#e74c3c", linewidth=2, markersize=6,
        label="Strategy A: Individual commits",
    )
    ax.plot(
        df["N"], df["strategy_b_per_record_ms"],
        "s-", color="#2ecc71", linewidth=2, markersize=6,
        label="Strategy B: Merkle batch",
    )

    # Mark crossover
    crossover_rows = df[df["crossover"]]
    if not crossover_rows.empty:
        cx = crossover_rows.iloc[0]
        ax.axvline(
            cx["N"], color="#7f8c8d", linestyle="--", linewidth=1, alpha=0.7,
        )
        ax.annotate(
            f"Crossover N={int(cx['N'])}",
            xy=(cx["N"], cx["strategy_a_per_record_ms"]),
            xytext=(cx["N"] * 1.5, cx["strategy_a_per_record_ms"] * 1.3),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
            fontsize=9, color="#7f8c8d",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (N compounds)", fontsize=12)
    ax.set_ylabel("Per-Record Anchoring Latency (ms)", fontsize=12)
    title = "Blockchain Anchoring Scalability: Individual vs. Merkle Batch"
    if offline:
        title += " [OFFLINE SIMULATION]"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    fig_path = os.path.join(results_dir, "scalability_figure.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure -> {fig_path}")


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", type=str,
        default="10,50,100,500",
        help="Comma-separated batch sizes (default: 10,50,100,500). "
             "Use 10,50,100,500,1000,5000,10000 for full sweep.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("PureProtX Blockchain Scalability Benchmark")
    print("=" * 60)
    df = benchmark_anchoring_scalability(batch_sizes=batch_sizes, n_repeats=args.repeats)
    print("\nResults:")
    print(df.to_string(index=False))
