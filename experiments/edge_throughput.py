#!/usr/bin/env python3
"""
B1 — Real-device edge-throughput harness (Reviewer R2.3).

Replaces the simulated Docker-resource-cap tiers (Table II / Fig. 4) with MEASURED
numbers on real hardware: x86 server baseline, Jetson Orin Nano, Raspberry Pi 4.

Pipeline per batch (compute-only; on-chain commit is separable and already
characterized in V-H, so it is deliberately excluded here):
  SMILES --RDKit--> 2058 features (10 descriptors + 2048 Morgan r=2)
        --scaler.onnx--> scaled --regressor/classifier ONNX--> consensus
        --canonical JSON--> SHA-256

Featurization is the EXACT one from pureprot/ai_model.py (copied here so the
harness is self-contained on the devices). Metrics per batch size: per-stage
latency (featurize / infer / hash), throughput (compounds/s), peak RSS memory.

Deps: rdkit, onnxruntime, numpy. (ORT 1.19.2 + numpy 2 + rdkit 2025.9.4 on the
devices — A1 showed ORT 1.19.2 output is byte-identical to the pinned 1.18.0.)

Run:
  python experiments/edge_throughput.py --csv chembl243_prepared_data.csv \
      --onnx-dir models/onnx --device x86 --out results/edge/edge_x86.json
"""

import argparse
import hashlib
import json
import os
import platform
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

N_FEATURES = 2058
REG = ["reg_svr", "reg_random_forest", "reg_gradient_boosting"]
CLF = ["clf_svc", "clf_rf_clf", "clf_gb_clf"]


def featurize(smiles):
    """EXACT featurization from pureprot/ai_model.py: 10 descriptors + 2048 Morgan."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.NumSaturatedRings(mol),
        # rdkit 2023+ renamed FractionCsp3 -> FractionCSP3; support both
        (getattr(Descriptors, "FractionCSP3", None) or Descriptors.FractionCsp3)(mol),
        Descriptors.BalabanJ(mol),
    ]
    feats.extend(list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
    return feats


def commit_digest(digest_hex, batch_id, rpc, contract_addr, abi, key, chain_id=900520900520):
    """Anchor a 32-byte digest on PureChain mainnet via recordScreeningResult.
    Returns commit latency (submit->receipt), tx hash, block number."""
    from web3 import Web3
    try:
        from web3.middleware import ExtraDataToPOAMiddleware
        poa = ExtraDataToPOAMiddleware
    except ImportError:
        from web3.middleware import geth_poa_middleware as poa
    w3 = Web3(Web3.HTTPProvider(rpc))
    w3.middleware_onion.inject(poa, layer=0)
    acct = w3.eth.account.from_key(key)
    c = w3.eth.contract(address=Web3.to_checksum_address(contract_addr), abi=abi)
    result_hash = bytes.fromhex(digest_hex)          # 32 bytes
    data_hash = hashlib.sha256(batch_id.encode()).digest()
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = c.functions.recordScreeningResult(result_hash, data_hash, batch_id).build_transaction(
        {"from": acct.address, "nonce": nonce, "gasPrice": 0, "chainId": chain_id, "gas": 500000})
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    t0 = time.perf_counter()
    txh = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(txh, timeout=90)
    return {"commit_ms": round((time.perf_counter() - t0) * 1000, 1),
            "tx": txh.hex(), "block": int(receipt.blockNumber)}


def peak_rss_mb():
    """Process peak resident memory in MB (Linux/mac via resource; else None)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes
        return round(kb / 1024.0, 1) if platform.system() != "Darwin" else round(kb / 1048576.0, 1)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 1048576.0, 1)
        except Exception:
            return None


def load_sessions(onnx_dir):
    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = {}
    for name in ["scaler"] + REG + CLF:
        p = os.path.join(onnx_dir, f"{name}.onnx")
        if os.path.exists(p):
            sess[name] = ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])
    return sess


def run_batch(sess, X):
    """scaler -> regressors (consensus) + classifiers. Returns (consensus, digest)."""
    sname = sess["scaler"].get_inputs()[0].name
    scaled = sess["scaler"].run(None, {sname: X.astype(np.float32)})[0]
    preds = []
    for r in REG:
        if r in sess:
            iname = sess[r].get_inputs()[0].name
            preds.append(np.asarray(sess[r].run(None, {iname: scaled.astype(np.float32)})[0]).flatten())
    consensus = np.mean(preds, axis=0) if preds else np.zeros(len(X))
    for c in CLF:
        if c in sess:
            iname = sess[c].get_inputs()[0].name
            sess[c].run(None, {iname: scaled.astype(np.float32)})  # exercised for realistic cost
    # canonical serialization of ranked results + hash
    ranked = sorted(range(len(consensus)), key=lambda i: -float(consensus[i]))
    payload = json.dumps({"ranking": ranked, "scores": [round(float(consensus[i]), 6) for i in ranked]},
                         sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return consensus, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--onnx-dir", default="models/onnx")
    ap.add_argument("--device", required=True, help="label, e.g. x86 / jetson / pi4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-sizes", default="100,500,1000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--storage", default="unknown", help="SD / NVMe / SSD")
    ap.add_argument("--commit", action="store_true", help="anchor each batch digest on PureChain mainnet")
    ap.add_argument("--deploy-json", default="local_deployment_info.json", help="JSON holding the contract ABI")
    ap.add_argument("--rpc", default=os.environ.get("PURECHAIN_RPC_URL", "https://purechainnode.com"))
    ap.add_argument("--contract", default=os.environ.get("CONTRACT_ADDRESS", "0xb8eb74663c1297825b188D8454a469d02Cc7d56C"))
    ap.add_argument("--key", default=os.environ.get("TEST_PRIVATE_KEY"))
    args = ap.parse_args()

    abi = None
    if args.commit:
        dj = args.deploy_json if os.path.isabs(args.deploy_json) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.deploy_json)
        with open(dj) as f:
            abi = json.load(f).get("abi", [])
        if not args.key:
            raise SystemExit("--commit requires TEST_PRIVATE_KEY (env or --key)")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = args.csv if os.path.isabs(args.csv) else os.path.join(root, args.csv)
    onnx_dir = args.onnx_dir if os.path.isabs(args.onnx_dir) else os.path.join(root, args.onnx_dir)
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    # load SMILES
    smiles = []
    with open(csv) as f:
        header = f.readline().strip().split(",")
        si = header.index("smiles") if "smiles" in header else 0
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) > si and parts[si]:
                smiles.append(parts[si])
    import onnxruntime as ort
    import rdkit
    sess = load_sessions(onnx_dir)

    manifest = {
        "device": args.device,
        "storage": args.storage,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "rdkit": rdkit.__version__,
        "n_compounds_available": len(smiles),
    }
    print(f"[edge] device={args.device} {manifest['machine']} ORT {manifest['onnxruntime']} "
          f"rdkit {manifest['rdkit']} cpus={manifest['cpu_count']}")

    rows = []
    for n in batch_sizes:
        if n > len(smiles):
            print(f"  skip N={n} (only {len(smiles)} compounds)"); continue
        batch = smiles[:n]
        feat_ms = infer_ms = hash_ms = total_ms = float("inf")
        n_valid = 0
        last_digest = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            feats, valid = [], 0
            for s in batch:
                fv = featurize(s)
                if fv is not None:
                    feats.append(fv); valid += 1
            X = np.array(feats, dtype=np.float64)
            t1 = time.perf_counter()
            _, digest = run_batch(sess, X)
            last_digest = digest
            t2 = time.perf_counter()
            # hash timing is inside run_batch; approximate as small tail — measure separately:
            hb0 = time.perf_counter()
            hashlib.sha256(X.tobytes()).hexdigest()
            hb1 = time.perf_counter()
            fe = (t1 - t0) * 1000; inf = (t2 - t1) * 1000; ha = (hb1 - hb0) * 1000
            tot = fe + inf
            if tot < total_ms:
                feat_ms, infer_ms, hash_ms, total_ms, n_valid = fe, inf, ha, tot, valid
        thr = n_valid / (total_ms / 1000.0) if total_ms > 0 else 0
        row = {
            "N": n, "n_valid": n_valid,
            "featurize_ms": round(feat_ms, 1), "infer_ms": round(infer_ms, 1),
            "hash_ms": round(hash_ms, 3), "total_ms": round(total_ms, 1),
            "throughput_cps": round(thr, 1), "throughput_cpm": round(thr * 60, 0),
            "peak_rss_mb": peak_rss_mb(),
        }
        if args.commit and last_digest:
            try:
                cr = commit_digest(last_digest, f"{args.device}_N{n}", args.rpc, args.contract, abi, args.key)
                row.update({"commit_ms": cr["commit_ms"], "tx": cr["tx"], "block": cr["block"],
                            "end_to_end_ms": round(total_ms + cr["commit_ms"], 1)})
            except Exception as e:
                row["commit_error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
        extra = (f" commit={row.get('commit_ms','-')}ms blk={row.get('block','-')}"
                 if args.commit else "")
        print(f"  N={n:5} valid={n_valid:5} feat={row['featurize_ms']:8.1f}ms "
              f"infer={row['infer_ms']:7.1f}ms total={row['total_ms']:8.1f}ms "
              f"thr={row['throughput_cps']:7.1f} cps  peakRSS={row['peak_rss_mb']}MB{extra}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"manifest": manifest, "results": rows}, f, indent=2)
    print(f"[edge] wrote {out}")


if __name__ == "__main__":
    main()
