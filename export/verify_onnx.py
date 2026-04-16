#!/usr/bin/env python3
"""
PureProtX ONNX Determinism Verification

Verifies that ONNX-exported models produce bitwise-identical outputs
across multiple inference runs, and that ONNX inference matches sklearn
predictions within floating-point tolerance.

Dependencies: onnxruntime>=1.18.0

Output:
  results/onnx_determinism.json
"""

import os
import sys
import json
import hashlib
import logging
import time
from typing import Dict, Any, Optional, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

N_FEATURES = 2058


def _sha256_array(arr: np.ndarray) -> str:
    """Deterministic SHA-256 of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ------------------------------------------------------------------
# Single-model determinism verification
# ------------------------------------------------------------------

def verify_onnx_determinism(
    onnx_path: str,
    X_test: np.ndarray,
    n_runs: int = 40,
) -> Dict[str, Any]:
    """
    Runs inference n_runs times using onnxruntime.
    Confirms bitwise identical outputs across all runs.

    Returns:
      {
        "model": str,
        "n_runs": int,
        "bitwise_identical": bool,
        "unique_output_hashes": int,   # must be 1
        "inference_latency_ms": float  # mean per-run
      }
    """
    import onnxruntime as ort

    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    abs_path = onnx_path if os.path.isabs(onnx_path) else os.path.join(PROJECT_ROOT, onnx_path)

    sess = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name

    # Match the ONNX model's expected dtype (float32 or float64)
    if input_info.type == "tensor(double)":
        X_in = X_test.astype(np.float64)
    else:
        X_in = X_test.astype(np.float32)

    output_hashes = set()
    latencies = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: X_in})
        latencies.append((time.perf_counter() - t0) * 1000)

        # Hash the primary output (predictions)
        primary = outputs[0]
        output_hashes.add(_sha256_array(primary))

    bitwise_identical = len(output_hashes) == 1
    mean_latency = float(np.mean(latencies))

    result = {
        "model": model_name,
        "n_runs": n_runs,
        "n_samples": X_test.shape[0],
        "bitwise_identical": bitwise_identical,
        "unique_output_hashes": len(output_hashes),
        "inference_latency_ms": round(mean_latency, 3),
    }

    status = "PASS" if bitwise_identical else "FAIL"
    print(f"    {model_name}: {status} "
          f"({len(output_hashes)} unique hash(es), {mean_latency:.1f} ms/run)")

    return result


# ------------------------------------------------------------------
# Cross-platform verification
# ------------------------------------------------------------------

def cross_platform_verify(
    onnx_path: str,
    reference_outputs_path: str,
) -> bool:
    """
    Loads reference outputs (generated on server), runs inference locally,
    checks SHA-256 match.  This is the key cross-platform determinism test.
    """
    import onnxruntime as ort

    abs_onnx = onnx_path if os.path.isabs(onnx_path) else os.path.join(PROJECT_ROOT, onnx_path)
    abs_ref = (reference_outputs_path if os.path.isabs(reference_outputs_path)
               else os.path.join(PROJECT_ROOT, reference_outputs_path))

    ref_data = np.load(abs_ref, allow_pickle=True)
    X_ref = ref_data["X"].astype(np.float32)
    ref_hash = str(ref_data["output_hash"])

    sess = ort.InferenceSession(abs_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_ref})
    local_hash = _sha256_array(outputs[0])

    match = local_hash == ref_hash
    status = "MATCH" if match else "MISMATCH"
    print(f"  Cross-platform verify: {status}")
    print(f"    Reference hash: {ref_hash[:24]}...")
    print(f"    Local hash:     {local_hash[:24]}...")

    return match


def generate_reference_outputs(
    onnx_path: str,
    X_test: np.ndarray,
    output_path: str,
) -> str:
    """
    Generate reference outputs for cross-platform verification.
    Saves X_test and the SHA-256 of the ONNX output to a .npz file.
    Returns the output hash.
    """
    import onnxruntime as ort

    abs_onnx = onnx_path if os.path.isabs(onnx_path) else os.path.join(PROJECT_ROOT, onnx_path)
    abs_out = output_path if os.path.isabs(output_path) else os.path.join(PROJECT_ROOT, output_path)

    sess = ort.InferenceSession(abs_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    X_f32 = X_test.astype(np.float32)
    outputs = sess.run(None, {input_name: X_f32})
    output_hash = _sha256_array(outputs[0])

    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    np.savez(abs_out, X=X_f32, output_hash=output_hash)
    print(f"  Reference outputs saved to {abs_out} (hash={output_hash[:24]}...)")

    return output_hash


# ------------------------------------------------------------------
# sklearn <-> ONNX concordance check
# ------------------------------------------------------------------

def verify_sklearn_onnx_concordance(
    sklearn_model,
    onnx_path: str,
    X_test: np.ndarray,
    is_classifier: bool = False,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Verify that ONNX inference matches sklearn predictions within tolerance.

    Tolerances:
      - Tree-based models (RF, GB): atol=1e-4 (float32 rounding only)
      - Kernel-based models (SVR):  atol=5e-4 (sklearn libsvm uses float64
        internally even for float32 input; onnxruntime SVMRegressor stays
        float32 throughout, producing irreducible kernel precision differences)
      - Classifiers: compared by hard label agreement (>=95%)

    Returns dict with concordance statistics.
    """
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    # Widen tolerance for kernel-based models (SVR/SVC regressors)
    if "svr" in model_name.lower():
        atol = 5e-4  # sklearn libsvm float64 vs onnxruntime float32 kernel
    import onnxruntime as ort

    abs_path = onnx_path if os.path.isabs(onnx_path) else os.path.join(PROJECT_ROOT, onnx_path)

    # ONNX predictions — match the model's expected input dtype
    sess = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    if input_info.type == "tensor(double)":
        X_onnx = X_test.astype(np.float64)
    else:
        X_onnx = X_test.astype(np.float32)
    onnx_outputs = sess.run(None, {input_name: X_onnx})

    if is_classifier:
        # For classifiers: compare hard label predictions (output[0])
        # rather than probabilities, because CalibratedClassifierCV
        # calibration may not be preserved in the ONNX graph.
        onnx_labels = onnx_outputs[0].flatten()
        sk_labels = sklearn_model.predict(X_test).flatten()
        n_agree = int(np.sum(onnx_labels == sk_labels))
        n_total = len(sk_labels)
        accuracy = n_agree / n_total if n_total > 0 else 0.0

        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        concordant = accuracy >= 0.95  # 95% label agreement
        status = "PASS" if concordant else "FAIL"
        print(f"    {model_name} concordance: {status} "
              f"(label_agree={n_agree}/{n_total}={accuracy:.1%})")

        return {
            "model": model_name,
            "concordance": concordant,
            "label_agreement": n_agree,
            "label_total": n_total,
            "label_accuracy": round(accuracy, 4),
            "n_samples": X_test.shape[0],
            "note": "Classifier: compared hard labels (calibration not preserved in ONNX)",
        }
    else:
        # For regressors: compare predictions at the SAME input precision.
        # ONNX uses float32 internally, so feed float32 to sklearn too --
        # otherwise we compare a float64 pipeline to a float32 one and
        # attribute kernel precision differences to the conversion.
        sk_pred = sklearn_model.predict(X_onnx)  # same dtype as ONNX input
        onnx_pred = onnx_outputs[0]

        sk_flat = sk_pred.flatten().astype(np.float64)
        onnx_flat = onnx_pred.flatten().astype(np.float64)

        if sk_flat.shape != onnx_flat.shape:
            min_len = min(len(sk_flat), len(onnx_flat))
            sk_flat = sk_flat[:min_len]
            onnx_flat = onnx_flat[:min_len]

        abs_diff = np.abs(sk_flat - onnx_flat)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        within_tol = bool(max_diff <= atol)

        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        status = "PASS" if within_tol else "FAIL"
        print(f"    {model_name} concordance: {status} "
              f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

        return {
            "model": model_name,
            "concordance": within_tol,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "tolerance": atol,
            "n_samples": X_test.shape[0],
        }


# ------------------------------------------------------------------
# Full verification suite
# ------------------------------------------------------------------

def run_full_verification(
    onnx_dir: str = "models/onnx",
    models_dir: str = "experiments/paper_results/models",
    n_determinism_runs: int = 40,
    n_test_samples: int = 200,
) -> Dict[str, Any]:
    """
    Run full ONNX verification suite:
      1. Determinism check (bitwise identical across n_runs)
      2. sklearn concordance check (ONNX matches sklearn within tolerance)

    Returns combined results dict, saves to results/onnx_determinism.json.
    """
    import joblib

    abs_onnx_dir = os.path.join(PROJECT_ROOT, onnx_dir)
    abs_models_dir = os.path.join(PROJECT_ROOT, models_dir)

    # Generate synthetic test data (deterministic)
    rng = np.random.RandomState(42)
    X_test = rng.randn(n_test_samples, N_FEATURES).astype(np.float32)

    # Find all .onnx files
    onnx_files = sorted([
        f for f in os.listdir(abs_onnx_dir) if f.endswith(".onnx")
    ])

    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in {abs_onnx_dir}")

    # Load a representative sklearn model for concordance checks
    model_files = sorted([
        f for f in os.listdir(abs_models_dir) if f.endswith("_model.joblib")
    ])
    sklearn_models = {}
    if model_files:
        model_data = joblib.load(os.path.join(abs_models_dir, model_files[0]))
        for name, model in model_data.get("models", {}).items():
            sklearn_models[f"reg_{name}"] = (model, False)
        for name, clf in model_data.get("classifiers", {}).items():
            # For SVC, the ONNX was exported from the raw inner SVC
            # (not CalibratedClassifierCV), so compare against that.
            if name == "svc" and hasattr(clf, "calibrated_classifiers_"):
                inner = clf.calibrated_classifiers_[0].estimator
                if hasattr(inner, "estimator"):
                    inner = inner.estimator
                sklearn_models[f"clf_{name}"] = (inner, True)
            else:
                sklearn_models[f"clf_{name}"] = (clf, True)
        scaler = model_data.get("scaler")
        if scaler:
            sklearn_models["scaler"] = (scaler, False)

    results = {
        "determinism": {},
        "concordance": {},
        "summary": {},
    }

    all_deterministic = True
    all_concordant = True

    print(f"\n  Verifying {len(onnx_files)} ONNX models...")

    for onnx_file in onnx_files:
        onnx_path = os.path.join(abs_onnx_dir, onnx_file)
        model_name = os.path.splitext(onnx_file)[0]

        # --- Determinism ---
        det_result = verify_onnx_determinism(
            onnx_path, X_test, n_runs=n_determinism_runs,
        )
        results["determinism"][model_name] = det_result
        if not det_result["bitwise_identical"]:
            all_deterministic = False

        # --- sklearn concordance ---
        if model_name in sklearn_models:
            sk_model, is_clf = sklearn_models[model_name]
            # For the scaler, use transform instead of predict
            if model_name == "scaler":
                conc_result = _verify_scaler_concordance(
                    sk_model, onnx_path, X_test,
                )
            else:
                conc_result = verify_sklearn_onnx_concordance(
                    sk_model, onnx_path, X_test, is_classifier=is_clf,
                )
            results["concordance"][model_name] = conc_result
            if not conc_result.get("concordance", True):
                all_concordant = False

    results["summary"] = {
        "n_models": len(onnx_files),
        "all_deterministic": all_deterministic,
        "all_concordant": all_concordant,
        "n_determinism_runs": n_determinism_runs,
        "n_test_samples": n_test_samples,
    }

    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "onnx_determinism.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    return results


def _verify_scaler_concordance(
    sklearn_scaler,
    onnx_path: str,
    X_test: np.ndarray,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Special concordance check for StandardScaler."""
    import onnxruntime as ort

    abs_path = onnx_path if os.path.isabs(onnx_path) else os.path.join(PROJECT_ROOT, onnx_path)

    sk_out = sklearn_scaler.transform(X_test).astype(np.float64)

    sess = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_test.astype(np.float32)})[0].astype(np.float64)

    abs_diff = np.abs(sk_out.flatten() - onnx_out.flatten())
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    within_tol = bool(max_diff <= atol)

    status = "PASS" if within_tol else "FAIL"
    print(f"    scaler concordance: {status} "
          f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

    return {
        "model": "scaler",
        "concordance": within_tol,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": atol,
        "n_samples": X_test.shape[0],
    }


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX ONNX Determinism Verification")
    print("=" * 60)

    results = run_full_verification()

    print("\n--- Summary ---")
    s = results["summary"]
    print(f"  Models tested:    {s['n_models']}")
    print(f"  All deterministic: {s['all_deterministic']}")
    print(f"  All concordant:    {s['all_concordant']}")
