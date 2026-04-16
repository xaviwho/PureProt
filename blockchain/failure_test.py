#!/usr/bin/env python3
"""
PureProtX PoA² Consensus Resilience Test

Measures the real PureChain mainnet's PoA² consensus properties:
  1. Baseline per-transaction consensus latency (N=20 transactions)
  2. Hash commitment + on-chain verification cycle
  3. Long-term hash integrity (commit, wait, re-verify)
  4. Sustained throughput under back-to-back load (N=50)

All measurements use the live PureChain mainnet (chain ID 900520900520).

Output:
  results/failure_test.json
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Dict, Any, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def test_poa2_resilience() -> Dict[str, Any]:
    """
    Full PoA² resilience test against real PureChain mainnet.

    Returns:
      {
        "chain_id": int,
        "rpc_url": str,
        "baseline_latencies": {median_ms, p95_ms, min_ms, max_ms, n},
        "sustained_latencies": {median_ms, p95_ms, min_ms, max_ms, n},
        "hash_integrity": {committed, verified, integrity_pct},
        "block_range": {first, last},
        "long_term_verify": {committed_block, verify_delay_s, verified_ok},
      }
    Saves results to results/failure_test.json
    """
    from blockchain.purechain_factory import get_purechain_connector

    print("  [1/5] Connecting to PureChain mainnet...")
    connector = get_purechain_connector(strict=True)
    chain_id = connector.w3.eth.chain_id
    start_block = connector.w3.eth.block_number
    print(f"    chain_id={chain_id}  block={start_block}  "
          f"wallet={connector.wallet_address}")

    results = {
        "chain_id": chain_id,
        "rpc_url": connector.rpc_url,
        "wallet": connector.wallet_address,
    }

    # Query the live validator set via Clique API
    print("  [1b/5] Querying PoA² validator set...")
    try:
        signers_resp = connector.w3.provider.make_request("clique_getSigners", ["latest"])
        validators = signers_resp.get("result", [])
        peers_resp = connector.w3.provider.make_request("admin_peers", [])
        n_peers = len(peers_resp.get("result", []))
        results["validators"] = {
            "count": len(validators),
            "addresses": validators,
            "connected_peers": n_peers,
        }
        print(f"    {len(validators)} active validators, {n_peers} connected peers")
        for v in validators:
            print(f"      {v}")
    except Exception as e:
        logger.warning("Could not query validator set: %s", e)
        results["validators"] = {"error": str(e)}

    # ----------------------------------------------------------
    # Phase 1: Baseline consensus latency (20 txs)
    # ----------------------------------------------------------
    print("  [2/5] Measuring baseline consensus latency (20 txs)...")
    baseline_latencies, baseline_txs = _measure_latency(connector, n=20, prefix="baseline")
    results["baseline_latencies"] = _latency_stats(baseline_latencies)
    print(f"    median={results['baseline_latencies']['median_ms']:.0f} ms  "
          f"p95={results['baseline_latencies']['p95_ms']:.0f} ms")

    # ----------------------------------------------------------
    # Phase 2: Hash integrity verification
    # ----------------------------------------------------------
    print("  [3/5] Verifying hash integrity (re-read committed txs)...")
    verified = 0
    for tx_hash, orig_hash in baseline_txs:
        vr = connector.verify_result_client_side(tx_hash, orig_hash)
        if vr.get("verified"):
            verified += 1
    results["hash_integrity"] = {
        "committed": len(baseline_txs),
        "verified": verified,
        "integrity_pct": round(verified / len(baseline_txs) * 100, 1) if baseline_txs else 0,
    }
    print(f"    {verified}/{len(baseline_txs)} verified (100% expected)")

    # ----------------------------------------------------------
    # Phase 3: Sustained load (50 rapid-fire txs)
    # ----------------------------------------------------------
    print("  [4/5] Sustained throughput under load (50 txs)...")
    sustained_latencies, sustained_txs = _measure_latency(connector, n=50, prefix="sustained")
    results["sustained_latencies"] = _latency_stats(sustained_latencies)
    print(f"    median={results['sustained_latencies']['median_ms']:.0f} ms  "
          f"p95={results['sustained_latencies']['p95_ms']:.0f} ms")

    # ----------------------------------------------------------
    # Phase 5: Block signer rotation analysis
    # ----------------------------------------------------------
    print("  [5/5] Analysing block signer rotation...")
    try:
        end_block = connector.w3.eth.block_number
        snapshot = connector.w3.provider.make_request(
            "clique_getSnapshot", [hex(end_block)])
        recents = snapshot.get("result", {}).get("recents", {})
        # recents maps block_number -> signer_address
        unique_signers = set(recents.values())
        results["signer_rotation"] = {
            "blocks_analysed": len(recents),
            "unique_signers": len(unique_signers),
            "signer_addresses": sorted(unique_signers),
        }
        print(f"    {len(unique_signers)} unique signers across "
              f"{len(recents)} recent blocks")
    except Exception as e:
        logger.warning("Signer rotation query failed: %s", e)
        results["signer_rotation"] = {"error": str(e)}

    # Long-term verify: check the very first baseline tx is still readable
    if baseline_txs:
        first_tx, first_hash = baseline_txs[0]
        vr = connector.verify_result_client_side(first_tx, first_hash)
        end_block = connector.w3.eth.block_number
        results["long_term_verify"] = {
            "first_tx": first_tx,
            "verified_ok": vr.get("verified", False),
            "blocks_elapsed": end_block - start_block,
        }

    results["block_range"] = {
        "first": start_block,
        "last": connector.w3.eth.block_number,
    }

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "failure_test.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {json_path}")

    return results


def _measure_latency(connector, n: int, prefix: str):
    """Submit n hashes and return (latencies_ms, [(tx_hash, orig_hash)])."""
    latencies = []
    txs = []
    for i in range(n):
        payload = f"{prefix}_{i}_{time.time()}".encode()
        result_hash = hashlib.sha256(payload).digest()
        data_hash = hashlib.sha256(f"data_{prefix}_{i}".encode()).digest()

        t0 = time.perf_counter()
        r = connector.record_and_verify_result(result_hash, data_hash, f"{prefix}_{i}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if r.get("success"):
            txs.append((r["tx_hash"], result_hash))
    return latencies, txs


def _latency_stats(latencies) -> dict:
    if not latencies:
        return {"median_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0, "n": 0}
    a = np.array(latencies)
    return {
        "n": len(a),
        "median_ms": round(float(np.median(a)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "min_ms": round(float(np.min(a)), 1),
        "max_ms": round(float(np.max(a)), 1),
        "mean_ms": round(float(np.mean(a)), 1),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX PoA2 Consensus Resilience Test")
    print("=" * 60)

    results = test_poa2_resilience()

    print("\n--- Summary ---")
    bl = results["baseline_latencies"]
    sl = results["sustained_latencies"]
    hi = results["hash_integrity"]
    print(f"  Baseline (20 txs):  median={bl['median_ms']:.0f} ms  p95={bl['p95_ms']:.0f} ms")
    print(f"  Sustained (50 txs): median={sl['median_ms']:.0f} ms  p95={sl['p95_ms']:.0f} ms")
    print(f"  Hash integrity:     {hi['verified']}/{hi['committed']} = {hi['integrity_pct']}%")
    br = results["block_range"]
    print(f"  Block range:        {br['first']} -> {br['last']}")
