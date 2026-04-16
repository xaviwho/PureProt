#!/usr/bin/env python3
"""
PureProtX Edge Workload Script

A minimal self-contained screening workload designed to run inside a
resource-constrained Docker container.  Loads a pre-trained joblib model,
generates N synthetic compounds (deterministic), runs the consensus
prediction pipeline, hashes the result, and commits the digest to PureChain.

Emits one machine-readable line on stdout when finished:
  EDGE_RESULT_JSON: {"latency_s": ..., "peak_mem_mb": ..., ...}

Usage (inside container):
  python -m edge.edge_workload --target CHEMBL243 --n 1000
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import List

import numpy as np

# ---------- cross-platform peak memory ----------

def _peak_rss_mb() -> float:
    """Peak resident set size in MB, works on Linux and Windows."""
    try:
        import resource  # type: ignore
        # On Linux, ru_maxrss is in kilobytes; on macOS it's bytes
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss_kb / (1024 * 1024)
        return rss_kb / 1024
    except ImportError:
        # Windows fallback via psutil if available, else 0
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except Exception:
            return 0.0


# ---------- pipeline ----------

def _generate_synthetic_features(n: int, seed: int = 42) -> np.ndarray:
    """Deterministic 2058-dim feature matrix (10 desc + 2048 fp bits)."""
    rng = np.random.RandomState(seed)
    # First 10 columns: descriptor-like floats
    desc = rng.randn(n, 10).astype(np.float32)
    # Last 2048 columns: binary fingerprint bits
    fp = (rng.rand(n, 2048) > 0.85).astype(np.float32)
    return np.hstack([desc, fp])


# Map targets to their study CSV files (relative to PROJECT_ROOT)
_TARGET_CSV_MAP = {
    "CHEMBL243":  "chembl243_prepared_data.csv",
    "CHEMBL247":  "chembl247_data.csv",
    "CHEMBL279":  "chembl279_prepared_data.csv",
    "CHEMBL3471": "chembl3471_data.csv",
    "CHEMBL2487": "modeling/data/chembl_2487_data.csv",
    "CHEMBL251":  "modeling/data/chembl251_data.csv",
    "CHEMBL217":  "modeling/data/chembl217_data.csv",
    "CHEMBL1862": "modeling/data/chembl1862_data.csv",
    "CHEMBL4005": "modeling/data/chembl4005_data.csv",
    "CHEMBL240":  "modeling/data/chembl240_data.csv",
}


def _load_study_smiles(target: str, n: int, seed: int = 42) -> list:
    """Load real SMILES from the study's ChEMBL dataset for this target.

    Samples N compounds (with replacement if N > dataset size) from
    the actual study CSV. Falls back to a minimal pool if the CSV
    is unavailable inside the container.
    """
    import pandas as pd

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_rel = _TARGET_CSV_MAP.get(target)
    rng = np.random.RandomState(seed)

    if csv_rel:
        csv_path = os.path.join(project_root, csv_rel)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["smiles"])
            pool = df["smiles"].dropna().tolist()
            if pool:
                idx = rng.randint(0, len(pool), size=n)
                return [pool[i] for i in idx]

    # Fallback: use a small set of study-representative drug-like SMILES
    fallback = [
        "CCC(C)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)NC(Cc1ccccc1)C(=O)O",
        "Cc1cc(C)c(/C=C2\\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)c32)[nH]1",
        "O=C1NCCN1CCN1CCC(c2cn(-c3ccc(F)cc3)c3ccc(Cl)cc23)CC1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    idx = rng.randint(0, len(fallback), size=n)
    return [fallback[i] for i in idx]


def _featurize_rdkit(n: int, seed: int = 42, target: str = "CHEMBL243") -> np.ndarray:
    """
    Compute real RDKit Morgan fingerprints + descriptors for N compounds
    drawn from the study's ChEMBL dataset for this target.

    This is the CPU-bound step that drives tier differentiation.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

    RDLogger.DisableLog("rdApp.*")

    smiles_list = _load_study_smiles(target, n, seed)
    features = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles("c1ccccc1")  # fallback

        # 10 molecular descriptors (same as ConsensusAIModel)
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            getattr(Descriptors, "FractionCSP3", getattr(Descriptors, "FractionCsp3", lambda m: 0.0))(mol),
            Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0,
        ]

        # 2048-bit Morgan fingerprint (radius=2)
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        desc.extend(list(fp))

        features.append(desc)

    return np.array(features, dtype=np.float32)


def run_workload(target: str, n_compounds: int, use_rdkit: bool = True) -> dict:
    """Run the full edge workload and return metrics.

    Args:
        use_rdkit: If True (default), compute real Morgan fingerprints + descriptors
                   via RDKit (CPU-bound). If False, use synthetic random features.
    """
    import joblib

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(
        project_root, "experiments", "paper_results", "models",
        f"{target}_model.joblib",
    )

    t_total_start = time.perf_counter()

    # 1. Load model (one-time cost amortised across screen)
    t0 = time.perf_counter()
    model_data = joblib.load(model_path)
    scaler = model_data["scaler"]
    models = model_data["models"]
    load_ms = (time.perf_counter() - t0) * 1000

    # 2. Featurize compounds
    t0 = time.perf_counter()
    if use_rdkit:
        X = _featurize_rdkit(n_compounds, target=target)
    else:
        X = _generate_synthetic_features(n_compounds)
    X_scaled = scaler.transform(X)
    featurise_ms = (time.perf_counter() - t0) * 1000

    # 3. Run consensus inference
    t0 = time.perf_counter()
    preds = np.mean(
        [m.predict(X_scaled) for m in models.values()], axis=0
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    # 4. Canonical JSON + SHA-256
    t0 = time.perf_counter()
    canonical = json.dumps(
        {"target": target, "n": n_compounds,
         "predictions": [round(float(p), 4) for p in preds]},
        sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    hash_ms = (time.perf_counter() - t0) * 1000

    # 5. Commit to PureChain
    t0 = time.perf_counter()
    bc_result = _commit_digest(digest, f"edge_{target}_{n_compounds}")
    bc_ms = (time.perf_counter() - t0) * 1000
    bc_success = bool(bc_result.get("success"))

    total_s = time.perf_counter() - t_total_start
    peak_mem_mb = _peak_rss_mb()
    throughput_cpm = (n_compounds / total_s) * 60 if total_s > 0 else 0.0

    return {
        "target": target,
        "n_compounds": n_compounds,
        "latency_s": round(total_s, 3),
        "model_load_ms": round(load_ms, 1),
        "featurise_ms": round(featurise_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "hash_ms": round(hash_ms, 1),
        "blockchain_latency_ms": round(bc_ms, 1),
        "blockchain_success": bc_success,
        "peak_mem_mb": round(peak_mem_mb, 2),
        "throughput_cpm": round(throughput_cpm, 1),
        "result_digest": digest,
        "block_number": bc_result.get("block_number", 0),
    }


def _commit_digest(digest_hex: str, mol_id: str) -> dict:
    """Commit a digest to PureChain via the shared connector factory."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from blockchain.purechain_factory import get_purechain_connector
        connector = get_purechain_connector(strict=False)
        if connector is None:
            return {"success": False, "offline": True}
        result_hash = bytes.fromhex(digest_hex)
        data_hash = hashlib.sha256(mol_id.encode()).digest()
        return connector.record_and_verify_result(result_hash, data_hash, mol_id)
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="CHEMBL243")
    parser.add_argument("--n", type=int, default=1000, dest="n_compounds")
    parser.add_argument("--no-rdkit", action="store_true",
                        help="Skip RDKit featurization (use synthetic features)")
    args = parser.parse_args()

    metrics = run_workload(args.target, args.n_compounds, use_rdkit=not args.no_rdkit)

    # Machine-readable line for the host profiler to parse
    print("EDGE_RESULT_JSON:", json.dumps(metrics))

    # Human-readable summary
    print(f"  target={metrics['target']}  n={metrics['n_compounds']}")
    print(f"  total      {metrics['latency_s']:>8.2f} s")
    print(f"  inference  {metrics['inference_ms']:>8.1f} ms")
    print(f"  blockchain {metrics['blockchain_latency_ms']:>8.1f} ms "
          f"(success={metrics['blockchain_success']})")
    print(f"  throughput {metrics['throughput_cpm']:>8.1f} cpm")
    print(f"  peak mem   {metrics['peak_mem_mb']:>8.1f} MB")


if __name__ == "__main__":
    main()
