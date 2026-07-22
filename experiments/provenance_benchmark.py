#!/usr/bin/env python3
"""
PureProtX Provenance Comparison Benchmark (Reviewer Comment R2.5)

Measures the LOCAL provenance baselines (Ed25519 signed append-only log,
IPFS-style content addressing, Merkle batch) implemented in
blockchain/provenance_baselines.py, and places them beside the ALREADY-MEASURED
PoA-squared / PureChain figures (reused from IOT_EXPERIMENT_RESULTS.md; NOT
re-measured here -- that would touch live mainnet).

The point (per R2.5) is NOT to show PoA-squared is fastest -- it is not, by
orders of magnitude. The point is to make the real trade-off explicit: local
mechanisms win decisively on latency/cost; PoA-squared wins on decentralised
trust (no single key holder), Byzantine fault tolerance, and independent public
verifiability. Both facts go in the table, unvarnished.

Outputs:
  results/provenance_comparison.csv    -- per-mechanism, per-N timing (local)
  results/provenance_comparison.md     -- qualitative + quantitative table
  results/provenance_manifest.json     -- versions + platform for reproducibility

Run:
  python -m experiments.provenance_benchmark
"""

import csv
import json
import os
import platform
import sys
import time
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from blockchain.provenance_baselines import (  # noqa: E402
    Ed25519SignedLog,
    _sha256,
    canonical_json,
    compute_cid_v1_raw,
    compute_merkle_root,
    merkle_proof,
    verify_cid_v1_raw,
    verify_merkle_proof,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BATCH_SIZES = [10, 100, 1000, 10000]
REPEATS = 5

# ------------------------------------------------------------------
# PoA-squared / PureChain figures REUSED from IOT_EXPERIMENT_RESULTS.md.
# These are NOT measured by this script (no mainnet contact). Source lines
# are cited so the provenance of these numbers is itself traceable.
# ------------------------------------------------------------------
POA2_REUSED = {
    "individual_commit_ms_per_record": 1904.0,   # V-C baseline median consensus latency
    "merkle_batch_ms_per_record_at_N100": 21.2,  # V-E Strategy B, N=100
    "source": "IOT_EXPERIMENT_RESULTS.md V-C (consensus) and V-E (scalability)",
    "note": "Real PureChain mainnet, 4-validator PoA2, ~2s consensus round-trip.",
}


def _deterministic_payload(i: int) -> Dict:
    """A deterministic synthetic screening record (no RNG, fully reproducible)."""
    return {
        "index": i,
        "smiles": f"C{'C' * (i % 12)}O",
        "target": "CHEMBL243",
        "pred_pic50": round(4.0 + (i % 600) / 100.0, 4),
    }


def _time_block(fn, repeats: int = REPEATS) -> float:
    """Return the best (min) wall-clock seconds over `repeats` runs of fn()."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark_local(n: int) -> Dict[str, float]:
    """Benchmark the three local mechanisms at batch size n. Returns ms/record."""
    payloads = [_deterministic_payload(i) for i in range(n)]
    canon = [canonical_json(p) for p in payloads]
    leaves = [_sha256(c) for c in canon]

    # --- Ed25519 signed append-only log ---
    def build_log():
        log = Ed25519SignedLog()
        for p in payloads:
            log.append(p)
        return log

    log_build_s = _time_block(build_log)
    log = build_log()
    log_verify_s = _time_block(lambda: log.verify())

    # --- IPFS-style content addressing (local CIDv1 raw) ---
    def build_cids():
        return [compute_cid_v1_raw(c) for c in canon]

    cid_build_s = _time_block(build_cids)
    cids = build_cids()
    cid_verify_s = _time_block(
        lambda: all(verify_cid_v1_raw(c, cid) for c, cid in zip(canon, cids))
    )

    # --- Merkle batch (one root over n leaves) ---
    merkle_build_s = _time_block(lambda: compute_merkle_root(leaves))
    root = compute_merkle_root(leaves)
    # Single-record inclusion verification (the realistic per-record check):
    proof0 = merkle_proof(leaves, 0)
    merkle_verify_one_s = _time_block(
        lambda: verify_merkle_proof(leaves[0], proof0, root)
    )

    return {
        "N": n,
        "ed25519_write_ms_per_rec": log_build_s / n * 1000,
        "ed25519_verify_ms_per_rec": log_verify_s / n * 1000,
        "cid_write_ms_per_rec": cid_build_s / n * 1000,
        "cid_verify_ms_per_rec": cid_verify_s / n * 1000,
        "merkle_root_ms_per_rec": merkle_build_s / n * 1000,
        "merkle_proof_len": len(proof0),
        "merkle_verify_one_ms": merkle_verify_one_s * 1000,
    }


def write_manifest() -> Dict:
    import cryptography
    manifest = {
        "experiment": "A3 provenance baselines (R2.5)",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cryptography_version": cryptography.__version__,
        "batch_sizes": BATCH_SIZES,
        "repeats_per_measurement": REPEATS,
        "timing": "best-of-N wall clock, single-thread, time.perf_counter",
        "storage_medium": "x86 dev box local SSD (not the Jetson SD card)",
        "note": (
            "Local mechanisms measured on this x86 host. PoA2/PureChain figures "
            "are reused, not re-measured; see provenance_comparison.md."
        ),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "provenance_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_csv(rows: List[Dict]) -> None:
    path = os.path.join(RESULTS_DIR, "provenance_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: List[Dict], manifest: Dict) -> None:
    r100 = next(r for r in rows if r["N"] == 100)
    ed = r100["ed25519_write_ms_per_rec"]
    cid = r100["cid_write_ms_per_rec"]
    poa2_ind = POA2_REUSED["individual_commit_ms_per_record"]

    def speedup(a, b):
        return f"{b / a:,.0f}x" if a > 0 else "n/a"

    lines: List[str] = []
    lines.append("# A3 - Real Provenance Baselines vs PoA-squared (Reviewer R2.5)\n")
    lines.append(
        "Reviewer 2 (R2.5) flagged the original blockchain baseline (individual "
        "commits vs Merkle batch) as a strawman. This experiment compares "
        "PureChain PoA-squared anchoring against two *legitimate* provenance "
        "mechanisms -- an Ed25519 signed append-only log and IPFS-style content "
        "addressing -- plus Merkle batching, across both cost and trust axes.\n"
    )
    lines.append(
        f"Local mechanisms measured on `{manifest['platform']}`, "
        f"Python {manifest['python_version']}, "
        f"cryptography {manifest['cryptography_version']}, "
        f"best-of-{REPEATS} single-thread. PoA-squared figures are **reused** "
        f"from {POA2_REUSED['source']} (no mainnet contact in this run).\n"
    )

    # --- Quantitative write-latency table (per record) ---
    lines.append("## Write latency (per record)\n")
    lines.append("| Mechanism | Write latency / record | Basis |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Ed25519 signed log | {ed*1000:,.1f} us | measured, N=100 |"
    )
    lines.append(
        f"| IPFS CID (content address, local) | {cid*1000:,.1f} us | measured, N=100 |"
    )
    lines.append(
        f"| Merkle root build | {r100['merkle_root_ms_per_rec']*1000:,.1f} us | measured, N=100 |"
    )
    lines.append(
        f"| PoA-squared individual commit | {poa2_ind:,.0f} ms | reused (V-C) |"
    )
    lines.append(
        f"| PoA-squared Merkle-batch (amortised) | "
        f"{POA2_REUSED['merkle_batch_ms_per_record_at_N100']:,.1f} ms | reused (V-E, N=100) |\n"
    )
    lines.append(
        f"At N=100, the Ed25519 log is ~**{speedup(ed, poa2_ind)}** faster per "
        f"record than an individual PoA-squared commit, and local content "
        f"addressing ~**{speedup(cid, poa2_ind)}** faster. **PoA-squared does "
        f"not win on latency and we do not claim it does.**\n"
    )

    # --- Qualitative trust/property matrix ---
    lines.append("## Trust & property matrix\n")
    lines.append(
        "| Property | Ed25519 signed log | IPFS content addressing | Merkle batch | PoA-squared (PureChain) |"
    )
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| Trust model | Single key holder (centralised) | Publisher chooses authoritative CID | Inherits its anchor's trust | Multi-validator consortium |"
    )
    lines.append(
        "| Fault tolerance | No | No | No (it is a data structure) | Tolerates minority validator faults; majority-honest authority set (NOT classical 1/3-Byzantine BFT) -- to be empirically tested in A4 |"
    )
    lines.append(
        "| Rewrite resistance | Key holder can re-sign history | None on its own (no ordering) | Needs an external anchor | Requires majority-of-validators collusion (no single party) |"
    )
    lines.append(
        "| Ordering + timestamp | Yes (chained) | No (content-addressed only) | No (unordered set) | Yes (block height + time) |"
    )
    lines.append(
        "| Independent public verification | Needs trusted public key | Needs trusted CID index | Needs the anchor | Public tx/state read via RPC (validator-set query disabled on current public endpoint) |"
    )
    lines.append(
        "| Tamper-evidence granularity | Per entry | Per object | Per leaf (log2 N proof) | Per committed hash |"
    )
    lines.append(
        "| Network / infra dependency | None (local) | IPFS network for persistence* | None | Live validator network |"
    )
    lines.append(
        "| Single point of compromise | The private key | The CID publisher | The anchor | None (distributed) |\n"
    )
    lines.append(
        "*IPFS persistence/retrieval was **not** measured here; only the local "
        "content-addressing property (CIDv1 raw, sha2-256) was computed. See the "
        "honesty note in blockchain/provenance_baselines.py.\n"
    )

    # --- Honest interpretation ---
    lines.append("## Honest interpretation\n")
    lines.append(
        "- **Where PoA-squared loses:** latency and cost. A signed local log or "
        "content addressing is 4-5 orders of magnitude cheaper per record. If the "
        "only requirement were tamper-evidence under a trusted operator, an "
        "Ed25519 append-only log would be the correct, simpler choice.\n"
    )
    lines.append(
        "- **Where PoA-squared wins (the actual contribution):** it removes the "
        "trusted single party. The Ed25519 log's immutability collapses if the key "
        "holder is compromised or dishonest; IPFS content addressing gives no "
        "ordering, timestamp, or protection against a publisher swapping which CID "
        "is 'the' record; Merkle batching still needs *something* to anchor the "
        "root -- and that anchor is exactly what PoA-squared provides. PoA-squared "
        "delivers multi-validator, majority-honest, publicly verifiable ordering "
        "with no single point of key compromise (its fault-tolerance envelope is "
        "quantified empirically in A4, not assumed here).\n"
    )
    lines.append(
        "- **Consequence for the paper (R2.5):** the correct framing is not "
        "'PoA-squared is faster' but 'PoA-squared is the right choice *only* when "
        "the trust model forbids a single authoritative key holder and requires "
        "independent public verifiability; otherwise a signed log is cheaper.' The "
        "Merkle-batch result should be presented as an optimisation *within* the "
        "PoA-squared anchor, not as a comparison baseline.\n"
    )

    path = os.path.join(RESULTS_DIR, "provenance_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print("=== A3 Provenance Baseline Benchmark (R2.5) ===")
    manifest = write_manifest()
    print(f"Platform: {manifest['platform']}")
    print(f"Python {manifest['python_version']}, cryptography {manifest['cryptography_version']}")
    rows = []
    for n in BATCH_SIZES:
        print(f"  benchmarking N={n} ...", flush=True)
        rows.append(benchmark_local(n))
    write_csv(rows)
    write_markdown(rows, manifest)

    print("\n--- Per-record write latency (measured local vs reused PoA2) ---")
    r100 = next(r for r in rows if r["N"] == 100)
    print(f"  Ed25519 signed log : {r100['ed25519_write_ms_per_rec']*1000:8.1f} us/record")
    print(f"  IPFS CID (local)   : {r100['cid_write_ms_per_rec']*1000:8.1f} us/record")
    print(f"  Merkle root build  : {r100['merkle_root_ms_per_rec']*1000:8.1f} us/record")
    print(f"  PoA2 individual    : {POA2_REUSED['individual_commit_ms_per_record']:8.1f} ms/record  (reused, V-C)")
    print(f"  PoA2 Merkle-batch  : {POA2_REUSED['merkle_batch_ms_per_record_at_N100']:8.1f} ms/record  (reused, V-E N=100)")
    print("\nWrote:")
    print("  results/provenance_comparison.csv")
    print("  results/provenance_comparison.md")
    print("  results/provenance_manifest.json")


if __name__ == "__main__":
    main()
