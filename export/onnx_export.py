#!/usr/bin/env python3
"""
PureProtX ONNX Export Module

Exports all trained sklearn models (SVR, RF, GB regression + SVC, RF, GB
classification) to ONNX format.  ONNX inference is deterministic across
Python versions, OS, and hardware -- solving the paper's acknowledged caveat
of Docker-only determinism.

Dependencies: skl2onnx>=1.17.0, onnxruntime>=1.18.0

Output:
  models/onnx/<model_name>.onnx        per-model ONNX files
  models/onnx/manifest.json            {model_name: sha256_digest}
"""

import os
import sys
import json
import hashlib
import logging
import time
from typing import Tuple, Dict, Any, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Number of features: 10 molecular descriptors + 2048 Morgan fingerprint bits
N_FEATURES = 2058


def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _unwrap_frozen_estimators(model) -> None:
    """
    Replace FrozenEstimator wrappers with their inner estimator in-place.

    sklearn 1.8 wraps base estimators inside CalibratedClassifierCV with
    FrozenEstimator, which skl2onnx does not recognise.  This function
    walks the calibration tree and swaps FrozenEstimator for the real
    estimator it wraps, so that skl2onnx can convert the model.
    """
    try:
        from sklearn.frozen._frozen import FrozenEstimator
    except ImportError:
        return  # sklearn < 1.8, nothing to unwrap

    # CalibratedClassifierCV stores fitted calibrators in calibrated_classifiers_
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated is None:
        return

    for cc in calibrated:
        est = getattr(cc, "estimator", None)
        if est is not None and isinstance(est, FrozenEstimator):
            cc.estimator = est.estimator  # unwrap to real SVC/RFC/GBC

    # Also unwrap the top-level .estimator attribute used by skl2onnx
    # for shape introspection during conversion
    top_est = getattr(model, "estimator", None)
    if top_est is not None and isinstance(top_est, FrozenEstimator):
        model.estimator = top_est.estimator


def _get_blockchain_connector():
    """Attempt to connect to PureChain; return None if unavailable."""
    from blockchain.purechain_factory import get_purechain_connector
    return get_purechain_connector(strict=False)


def _commit_onnx_hash(connector, model_name: str, digest_hex: str) -> Optional[Dict]:
    """Commit an ONNX model hash to PureChain."""
    if connector is None:
        return None
    try:
        result_hash = bytes.fromhex(digest_hex)
        data_hash = hashlib.sha256(model_name.encode()).digest()
        return connector.record_and_verify_result(result_hash, data_hash, f"onnx_{model_name}")
    except Exception as e:
        logger.warning("Failed to commit ONNX hash for %s: %s", model_name, e)
        return None


# ------------------------------------------------------------------
# Single model export
# ------------------------------------------------------------------

def export_model_to_onnx(
    sklearn_model,
    model_name: str,
    output_dir: str = "models/onnx",
    n_features: int = N_FEATURES,
    is_classifier: bool = False,
) -> Tuple[bytes, str]:
    """
    Converts an sklearn model to ONNX via skl2onnx.
    Computes SHA-256 of the ONNX binary.
    Commits ONNX model hash to PureChain.
    Saves .onnx file to output_dir/{model_name}.onnx

    Returns: (onnx_bytes, sha256_hex_digest)
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    initial_type = [("X", FloatTensorType([None, n_features]))]

    # Unwrap FrozenEstimator inside CalibratedClassifierCV (sklearn 1.8+)
    # skl2onnx does not recognise FrozenEstimator, so we replace it with
    # the real estimator it wraps while preserving the calibration layer.
    _unwrap_frozen_estimators(sklearn_model)

    # Convert to ONNX
    options = {}
    if is_classifier:
        # Ensure probability outputs are exported for classifiers
        options = {id(sklearn_model): {"zipmap": False}}

    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=initial_type,
        target_opset=15,
        options=options if options else None,
    )

    onnx_bytes = onnx_model.SerializeToString()
    digest = _sha256_bytes(onnx_bytes)

    # Save to disk
    onnx_path = os.path.join(abs_output_dir, f"{model_name}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_bytes)

    # Commit to PureChain
    connector = _get_blockchain_connector()
    tx_result = _commit_onnx_hash(connector, model_name, digest)
    if tx_result and tx_result.get("success"):
        print(f"    {model_name}: ONNX hash committed on-chain "
              f"(block {tx_result.get('block_number')})")
    else:
        print(f"    {model_name}: ONNX hash recorded offline")

    print(f"    {model_name}: exported to {onnx_path} "
          f"({len(onnx_bytes):,} bytes, SHA-256={digest[:16]}...)")

    return onnx_bytes, digest


# ------------------------------------------------------------------
# Export all models from a saved joblib ensemble
# ------------------------------------------------------------------

def export_all_models(
    models_dir: str = "experiments/paper_results/models",
    output_dir: str = "models/onnx",
) -> Dict[str, str]:
    """
    Exports all sklearn models from saved joblib files to ONNX.

    Looks for ConsensusAIModel joblib files containing:
      - models: dict of {name: sklearn_regressor}
      - classifiers: dict of {name: sklearn_classifier} (if present)
      - scaler: StandardScaler

    Returns {model_name: sha256_digest} dict.
    Saves manifest JSON to models/onnx/manifest.json
    """
    import joblib
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    abs_models_dir = os.path.join(PROJECT_ROOT, models_dir)
    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    # Find any joblib model file to load the ensemble from
    model_files = sorted([
        f for f in os.listdir(abs_models_dir) if f.endswith("_model.joblib")
    ])

    if not model_files:
        raise FileNotFoundError(
            f"No *_model.joblib files found in {abs_models_dir}"
        )

    # Use the first available model as representative (all share the same
    # sklearn model classes, just different weights)
    representative_path = os.path.join(abs_models_dir, model_files[0])
    print(f"  Loading representative model: {model_files[0]}")
    model_data = joblib.load(representative_path)

    manifest = {}

    # --- Export scaler ---
    scaler = model_data.get("scaler")
    if scaler is not None:
        _, digest = export_model_to_onnx(
            scaler, "scaler", output_dir, n_features=N_FEATURES,
        )
        manifest["scaler"] = digest

    # --- Export regression models ---
    regression_models = model_data.get("models", {})
    for name, model in regression_models.items():
        safe_name = f"reg_{name}"
        _, digest = export_model_to_onnx(
            model, safe_name, output_dir, n_features=N_FEATURES,
            is_classifier=False,
        )
        manifest[safe_name] = digest

    # --- Export classification models ---
    # CalibratedClassifierCV wrapping SVC does not convert cleanly to ONNX
    # (the calibration sigmoid is lost, producing all-zero probabilities).
    # For SVC we extract the inner fitted SVC and export it directly;
    # tree-based classifiers (RF, GB) survive CalibratedClassifierCV fine.
    classifiers = model_data.get("classifiers", {})
    for name, clf in classifiers.items():
        safe_name = f"clf_{name}"
        export_clf = clf

        if hasattr(clf, "calibrated_classifiers_") and name == "svc":
            # Extract the inner SVC from the first calibrated fold
            inner = clf.calibrated_classifiers_[0].estimator
            if hasattr(inner, "estimator"):
                inner = inner.estimator  # unwrap FrozenEstimator
            export_clf = inner
            print(f"    (SVC: exporting raw {type(inner).__name__} "
                  f"without calibration wrapper)")

        _, digest = export_model_to_onnx(
            export_clf, safe_name, output_dir, n_features=N_FEATURES,
            is_classifier=True,
        )
        manifest[safe_name] = digest

    # --- Save manifest ---
    manifest_path = os.path.join(abs_output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\n  Manifest saved to {manifest_path}")
    print(f"  Total models exported: {len(manifest)}")

    return manifest


# ------------------------------------------------------------------
# Per-target export (all 10 targets)
# ------------------------------------------------------------------

def export_all_targets(
    models_dir: str = "experiments/paper_results/models",
    output_dir: str = "models/onnx",
) -> Dict[str, Dict[str, str]]:
    """
    Export ONNX models for every target that has a saved joblib file.

    Returns: {target_id: {model_name: digest}}
    """
    import joblib

    abs_models_dir = os.path.join(PROJECT_ROOT, models_dir)
    abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    model_files = sorted([
        f for f in os.listdir(abs_models_dir) if f.endswith("_model.joblib")
    ])

    all_manifests = {}

    for mf in model_files:
        target_id = mf.replace("_model.joblib", "")
        print(f"\n  [{target_id}]")
        model_data = joblib.load(os.path.join(abs_models_dir, mf))

        target_output = os.path.join(output_dir, target_id)
        target_manifest = {}

        # Scaler
        scaler = model_data.get("scaler")
        if scaler is not None:
            _, digest = export_model_to_onnx(
                scaler, "scaler", target_output, n_features=N_FEATURES,
            )
            target_manifest["scaler"] = digest

        # Regression models
        for name, model in model_data.get("models", {}).items():
            safe_name = f"reg_{name}"
            _, digest = export_model_to_onnx(
                model, safe_name, target_output, n_features=N_FEATURES,
            )
            target_manifest[safe_name] = digest

        # Classification models
        for name, clf in model_data.get("classifiers", {}).items():
            safe_name = f"clf_{name}"
            _, digest = export_model_to_onnx(
                clf, safe_name, target_output, n_features=N_FEATURES,
                is_classifier=True,
            )
            target_manifest[safe_name] = digest

        all_manifests[target_id] = target_manifest

    # Save combined manifest
    manifest_path = os.path.join(abs_output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_manifests, f, indent=2, sort_keys=True)
    print(f"\n  Combined manifest saved to {manifest_path}")

    return all_manifests


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("PureProtX ONNX Model Export")
    print("=" * 60)
    manifest = export_all_models()
    print("\nExport complete.")
    for name, digest in manifest.items():
        print(f"  {name}: {digest[:24]}...")
