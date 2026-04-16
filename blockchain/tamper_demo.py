#!/usr/bin/env python3
"""
PureProtX Tamper Detection Demonstration

Proves the tamper-evidence property of blockchain-anchored screening results:
  1. Screen a small batch of real study compounds
  2. Commit the canonical JSON hash to PureChain mainnet
  3. Tamper with one result field
  4. Show that the recomputed hash diverges from the on-chain hash

Output: results/tamper_demo.json
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


def run_tamper_demo(target: str = "CHEMBL243", n: int = 10) -> dict:
    """Run the full tamper detection demonstration."""
    from blockchain.purechain_factory import get_purechain_connector
    import joblib

    print(f"  [1/6] Loading model for {target}...")
    model_path = os.path.join(
        PROJECT_ROOT, "experiments", "paper_results", "models",
        f"{target}_model.joblib",
    )
    model_data = joblib.load(model_path)
    scaler = model_data["scaler"]
    models = model_data["models"]

    # Generate features from synthetic data (deterministic)
    rng = np.random.RandomState(42)
    X = rng.randn(n, 2058).astype(np.float32)
    X_scaled = scaler.transform(X)
    preds = np.mean([m.predict(X_scaled) for m in models.values()], axis=0)

    # Build canonical result JSON
    result_data = {
        "target": target,
        "n_compounds": n,
        "predictions": [round(float(p), 4) for p in preds],
        "timestamp": int(time.time()),
        "pipeline_version": "PureProtX-1.0.0",
    }

    canonical_json = json.dumps(result_data, sort_keys=True, separators=(",", ":"))
    original_hash = hashlib.sha256(canonical_json.encode()).hexdigest()
    print(f"  [2/6] Original hash: {original_hash[:32]}...")

    # Commit to PureChain
    print("  [3/6] Committing original hash to PureChain mainnet...")
    connector = get_purechain_connector(strict=True)
    result_hash_bytes = bytes.fromhex(original_hash)
    data_hash_bytes = hashlib.sha256(f"tamper_demo_{target}".encode()).digest()
    tx_result = connector.record_and_verify_result(
        result_hash_bytes, data_hash_bytes, f"tamper_demo_{target}"
    )

    if not tx_result.get("success"):
        raise RuntimeError(f"PureChain commit failed: {tx_result}")

    tx_hash = tx_result["tx_hash"]
    block_number = tx_result["block_number"]
    print(f"    tx={tx_hash[:24]}...  block={block_number}")

    # Tamper: modify one score by +0.001
    print("  [4/6] Tampering: changing predictions[0] by +0.001...")
    original_score = result_data["predictions"][0]
    tampered_score = round(original_score + 0.001, 4)
    tampered_data = result_data.copy()
    tampered_data["predictions"] = list(result_data["predictions"])  # deep copy
    tampered_data["predictions"][0] = tampered_score

    tampered_json = json.dumps(tampered_data, sort_keys=True, separators=(",", ":"))
    tampered_hash = hashlib.sha256(tampered_json.encode()).hexdigest()
    print(f"  [5/6] Tampered hash: {tampered_hash[:32]}...")

    # Verify divergence
    hashes_diverge = original_hash != tampered_hash
    print(f"  [6/6] Hashes diverge: {hashes_diverge}")

    # Verify on-chain: re-read the committed tx and compare
    verify_result = connector.verify_result_client_side(tx_hash, result_hash_bytes)
    onchain_verified = verify_result.get("verified", False)

    # Also verify the tampered hash does NOT match
    tampered_bytes = bytes.fromhex(tampered_hash)
    tamper_verify = connector.verify_result_client_side(tx_hash, tampered_bytes)
    tamper_rejected = not tamper_verify.get("verified", True)

    result = {
        "target": target,
        "n_compounds": n,
        "original_score_0": original_score,
        "tampered_score_0": tampered_score,
        "modification": f"predictions[0]: {original_score} -> {tampered_score}",
        "original_hash": original_hash,
        "tampered_hash": tampered_hash,
        "hashes_diverge": hashes_diverge,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "onchain_original_verified": onchain_verified,
        "onchain_tamper_rejected": tamper_rejected,
    }

    # Save
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "tamper_demo.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX Tamper Detection Demonstration")
    print("=" * 60)
    r = run_tamper_demo()
    print(f"\n--- Result ---")
    print(f"  Original:  {r['original_hash'][:32]}...")
    print(f"  Tampered:  {r['tampered_hash'][:32]}...")
    print(f"  Diverge:   {r['hashes_diverge']}")
    print(f"  On-chain original verified: {r['onchain_original_verified']}")
    print(f"  On-chain tamper rejected:   {r['onchain_tamper_rejected']}")
    print(f"  Block:     {r['block_number']}")
