#!/usr/bin/env python3
"""
A1 dtype axis — float32 vs float64 ONNX vs sklearn precision (Reviewer R2.4).

R2.4 specifically questions the dtype dimension of the determinism claim. This
quantifies, per regressor, how far ONNX inference sits from the sklearn reference
at float32 vs float64 export, on the same fixed seeded input as the harness.

Expected story (to be confirmed): tree ensembles + scaler are near-exact at both
dtypes; SVR (RBF kernel) shows a float32-only gap because onnxruntime's
SVMRegressor stays float32 while sklearn/libsvm computes the kernel in float64 —
and exporting SVR as float64 closes that gap. This is the concrete basis for the
A2 rescoping ("bitwise-deterministic within a pinned dtype/runtime config").

Needs: onnxruntime, scikit-learn, skl2onnx, joblib, numpy<2 (ORT 1.18 ABI).

Output: results/determinism/dtype_concordance.json
"""

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
N_FEATURES = 2058
SEED = 42
MODELS_DIR = os.path.join(ROOT, "experiments", "paper_results", "models")
OUT = os.path.join(ROOT, "results", "determinism", "dtype_concordance.json")


def _unwrap_frozen(model):
    try:
        from sklearn.frozen._frozen import FrozenEstimator
    except ImportError:
        return
    cal = getattr(model, "calibrated_classifiers_", None)
    if cal:
        for cc in cal:
            est = getattr(cc, "estimator", None)
            if isinstance(est, FrozenEstimator):
                cc.estimator = est.estimator


def export(est, dtype):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
    tt = DoubleTensorType if dtype == "float64" else FloatTensorType
    onx = convert_sklearn(est, initial_types=[("X", tt([None, N_FEATURES]))],
                          target_opset=15)
    return onx.SerializeToString()


def onnx_predict(onnx_bytes, X, dtype):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    Xi = X.astype(np.float64 if dtype == "float64" else np.float32)
    return np.asarray(sess.run(None, {name: Xi})[0]).flatten().astype(np.float64)


def main():
    import joblib
    mfiles = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith("_model.joblib"))
    data = joblib.load(os.path.join(MODELS_DIR, mfiles[0]))
    print(f"[dtype] representative model: {mfiles[0]}")

    rng = np.random.RandomState(SEED)
    X = rng.randn(64, N_FEATURES)

    targets = []
    for name, est in data.get("models", {}).items():
        targets.append((f"reg_{name}", est, "predict"))
    if data.get("scaler") is not None:
        targets.append(("scaler", data["scaler"], "transform"))

    results = {}
    import onnxruntime as ort
    for name, est, how in targets:
        _unwrap_frozen(est)
        # sklearn reference in float64
        ref = (est.predict(X) if how == "predict" else est.transform(X)).flatten().astype(np.float64)
        row = {}
        for dtype in ("float32", "float64"):
            try:
                b = export(est, dtype)
                pred = onnx_predict(b, X, dtype)
                n = min(len(pred), len(ref))
                row[dtype] = {
                    "exported": True,
                    "max_abs_diff_vs_sklearn": float(np.max(np.abs(pred[:n] - ref[:n]))),
                    "mean_abs_diff_vs_sklearn": float(np.mean(np.abs(pred[:n] - ref[:n]))),
                }
            except Exception as e:
                row[dtype] = {"exported": False, "error": f"{type(e).__name__}: {e}"[:200]}
        results[name] = row
        f32 = row["float32"].get("max_abs_diff_vs_sklearn")
        f64 = row["float64"].get("max_abs_diff_vs_sklearn")
        print(f"  {name:24} f32 max_diff={f32}  f64 max_diff={f64}")

    manifest = {
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "note": "ONNX vs sklearn(float64) max abs diff on fixed seed-42 input, n=64.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"manifest": manifest, "results": results}, f, indent=2)
    print(f"[dtype] wrote {OUT}")


if __name__ == "__main__":
    main()
