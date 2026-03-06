"""
Track 5: Hash Stability & Determinism Validation Suite

Empirically proves that classical ML models produce identical outputs across runs,
and validates the full pipeline hash determinism.
"""

import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hash_predictions(predictions, precision=4):
    """Hash an array of predictions to a SHA-256 digest."""
    rounded = [round(float(p), precision) for p in predictions]
    payload = json.dumps(rounded, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def generate_synthetic_data(n_samples=200, n_features=100, random_state=42):
    """Generate reproducible synthetic molecular feature data."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # Simulate pIC50 values
    y = 5.0 + X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.randn(n_samples) * 0.1
    return X, y


def test_model_determinism(n_runs=5):
    """
    Test that each model produces identical predictions across multiple runs
    with the same seed and data.

    Returns:
        Dict with model name -> {deterministic: bool, unique_hashes: int, hashes: list}
    """
    print("=" * 60)
    print("TRACK 5.1: Model Determinism Test")
    print("=" * 60)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regression models
    models_to_test = {
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(256, 128), random_state=42, max_iter=500),
    }

    # Try importing XGBoost
    try:
        from xgboost import XGBRegressor
        models_to_test['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
    except ImportError:
        print("  (XGBoost not installed, skipping)")

    results = {}

    for name, model in models_to_test.items():
        hashes = []
        for run in range(n_runs):
            model_copy = clone(model)
            model_copy.fit(X_train_scaled, y_train)
            preds = model_copy.predict(X_test_scaled)
            h = hash_predictions(preds)
            hashes.append(h)

        unique_count = len(set(hashes))
        is_deterministic = unique_count == 1

        results[name] = {
            'deterministic': is_deterministic,
            'unique_hashes': unique_count,
            'hashes': hashes
        }

        status = 'DETERMINISTIC' if is_deterministic else 'NON-DETERMINISTIC'
        print(f"  {name:20s}: {status} (unique hashes: {unique_count}/{n_runs})")

    return results


def test_classification_determinism(n_runs=5):
    """Test classification model determinism."""
    print("\n" + "=" * 60)
    print("TRACK 5.1b: Classification Model Determinism Test")
    print("=" * 60)

    X, y_cont = generate_synthetic_data()
    y_bin = (y_cont >= np.median(y_cont)).astype(int)
    X_train, X_test, y_train, _ = train_test_split(X, y_bin, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        'SVC': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'RF_clf': RandomForestClassifier(n_estimators=100, random_state=42),
        'GB_clf': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        hashes = []
        for run in range(n_runs):
            clf_copy = clone(clf)
            clf_copy.fit(X_train_scaled, y_train)
            probas = clf_copy.predict_proba(X_test_scaled)[:, 1]
            h = hash_predictions(probas)
            hashes.append(h)

        unique_count = len(set(hashes))
        is_deterministic = unique_count == 1

        results[name] = {
            'deterministic': is_deterministic,
            'unique_hashes': unique_count,
        }

        status = 'DETERMINISTIC' if is_deterministic else 'NON-DETERMINISTIC'
        print(f"  {name:20s}: {status} (unique hashes: {unique_count}/{n_runs})")

    return results


def test_hash_pipeline_determinism(n_runs=5):
    """
    Test that the canonical JSON + SHA-256 hash pipeline is deterministic.
    """
    print("\n" + "=" * 60)
    print("TRACK 5.1c: Hash Pipeline Determinism Test")
    print("=" * 60)

    # Simulate a screening result record
    record = {
        'molecule_id': 'CHEMBL25',
        'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'biomaterial_id': 'bio:cell_line:hela-001',
        'model_hash': 'abc123def456',
        'parameters_hash': '789ghi012jkl',
        'results': {
            'svr': 6.4,
            'random_forest': 6.6,
            'gradient_boosting': 6.5,
            'consensus': 6.5
        }
    }

    hashes = []
    for _ in range(n_runs):
        canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))
        h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        hashes.append(h)

    unique_count = len(set(hashes))
    is_deterministic = unique_count == 1

    status = 'DETERMINISTIC' if is_deterministic else 'NON-DETERMINISTIC'
    print(f"  Canonical JSON+SHA256: {status} (unique hashes: {unique_count}/{n_runs})")
    print(f"  Hash: {hashes[0][:16]}...")

    return {'deterministic': is_deterministic, 'hash': hashes[0]}


def test_consensus_determinism(n_runs=5):
    """
    Test full consensus ensemble pipeline determinism
    (train + predict + hash of predictions).
    """
    print("\n" + "=" * 60)
    print("TRACK 5.1d: Full Consensus Pipeline Determinism Test")
    print("=" * 60)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pipeline_hashes = []

    for run in range(n_runs):
        models = {
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        all_preds = []
        for model in models.values():
            m = clone(model)
            m.fit(X_train_scaled, y_train)
            preds = m.predict(X_test_scaled)
            all_preds.append(preds)

        consensus = np.mean(all_preds, axis=0)
        h = hash_predictions(consensus)
        pipeline_hashes.append(h)

    unique_count = len(set(pipeline_hashes))
    is_deterministic = unique_count == 1

    status = 'DETERMINISTIC' if is_deterministic else 'NON-DETERMINISTIC'
    print(f"  Full pipeline:       {status} (unique hashes: {unique_count}/{n_runs})")

    return {'deterministic': is_deterministic, 'unique_hashes': unique_count}


def run_all_determinism_tests():
    """Run complete determinism test suite and produce summary."""
    print("\n" + "#" * 60)
    print("# PureProtX Determinism Validation Suite")
    print("#" * 60)

    results = {}
    results['regression_models'] = test_model_determinism()
    results['classification_models'] = test_classification_determinism()
    results['hash_pipeline'] = test_hash_pipeline_determinism()
    results['consensus_pipeline'] = test_consensus_determinism()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for category, cat_results in results.items():
        if isinstance(cat_results, dict) and 'deterministic' in cat_results:
            status = 'PASS' if cat_results['deterministic'] else 'FAIL'
            if not cat_results['deterministic']:
                all_pass = False
            print(f"  {category}: {status}")
        else:
            for name, res in cat_results.items():
                status = 'PASS' if res['deterministic'] else 'FAIL'
                if not res['deterministic']:
                    all_pass = False
                print(f"  {category}/{name}: {status}")

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return results


if __name__ == '__main__':
    run_all_determinism_tests()
