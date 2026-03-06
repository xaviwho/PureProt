#!/usr/bin/env python3
"""Analysis of Vina scores with score capping and BEDROC optimization."""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pureprot.ai_model import ConsensusAIModel
from pureprot.evaluation import (
    normalize_ai_scores, normalize_docking_scores,
    optimize_alpha, evaluate_ranking
)

TARGETS = {
    'CHEMBL217': 'modeling/data/chembl217_data.csv',
    'CHEMBL251': 'modeling/data/chembl251_data.csv',
    'CHEMBL2487': 'modeling/data/chembl_2487_data.csv',
    'CHEMBL243': 'chembl243_prepared_data.csv',
    'CHEMBL1862': 'modeling/data/chembl1862_data.csv',
    'CHEMBL3471': 'chembl3471_data.csv',
    'CHEMBL4005': 'modeling/data/chembl4005_data.csv',
    'CHEMBL247': 'chembl247_data.csv',
}


def load_vina_capped(target_id, set_name, n_expected):
    """Load Vina scores with capping and median imputation."""
    cache_file = f'docking_cache/{target_id}_{set_name}_vina.json'
    if not os.path.exists(cache_file):
        return None
    with open(cache_file) as f:
        cached = json.load(f)
    scores = np.array(cached['scores'])
    if len(scores) != n_expected:
        return None

    # Cap outliers: any positive score or < -15 is clamped
    scores = np.clip(scores, -15.0, 0.0)

    # Median imputation for zeros (failed/capped)
    nz = scores != 0.0
    if np.sum(nz) > 0 and np.sum(~nz) > 0:
        scores[~nz] = np.median(scores[nz])

    return scores


for target_id, csv_file in TARGETS.items():
    print(f'=== {target_id} ===')
    model = ConsensusAIModel(model_type='both')
    model.train(data_path=csv_file, activity_threshold=6.0, activity_mode='threshold')
    _, _, smiles_list = model.prepare_dataset(csv_file, return_smiles=True)

    vd = model._val_data
    y_val, y_test = vd['y_val'], vd['y_test']
    X_val, X_test = vd['X_val_scaled'], vd['X_test_scaled']
    split = model.split_info
    val_idx, test_idx = split['val_indices'], split['test_indices']

    y_val_bin = model._make_binary_labels(y_val, threshold=6.0, mode='threshold')
    y_test_bin = model._make_binary_labels(y_test, threshold=6.0, mode='threshold')

    if y_val_bin is None or y_test_bin is None:
        print('  Insufficient binary labels')
        continue

    fAI_val = normalize_ai_scores(model.predict_consensus(X_val))
    fAI_test = normalize_ai_scores(model.predict_consensus(X_test))

    # Load capped Vina scores
    vina_val = load_vina_capped(target_id, 'val', len(val_idx))
    vina_test = load_vina_capped(target_id, 'test', len(test_idx))
    if vina_val is None or vina_test is None:
        print('  Vina cache not found or size mismatch')
        continue

    fdock_val = normalize_docking_scores(vina_val)
    fdock_test = normalize_docking_scores(vina_test)

    corr = np.corrcoef(fAI_test, fdock_test)[0, 1]

    # Regression baseline
    reg_m = evaluate_ranking(fAI_test, y_test_bin, 'reg')

    # BEDROC-optimized alpha
    best_alpha, best_val = optimize_alpha(fAI_val, fdock_val, y_val_bin, metric='bedroc')
    hybrid = best_alpha * fAI_test + (1 - best_alpha) * fdock_test
    hyb_m = evaluate_ranking(hybrid, y_test_bin, 'hybrid')

    delta_ef1 = hyb_m['ef1'] - reg_m['ef1']
    delta_bed = hyb_m['bedroc_20'] - reg_m['bedroc_20']
    delta_roc = hyb_m['auc_roc'] - reg_m['auc_roc']

    print(f'  Corr(AI,Vina)={corr:.3f}  alpha={best_alpha:.2f}')
    print(f'  Regression: EF1={reg_m["ef1"]:.2f}  BEDROC={reg_m["bedroc_20"]:.3f}  ROC={reg_m["auc_roc"]:.3f}')
    print(f'  Hybrid:     EF1={hyb_m["ef1"]:.2f}  BEDROC={hyb_m["bedroc_20"]:.3f}  ROC={hyb_m["auc_roc"]:.3f}')
    print(f'  Delta:      EF1={delta_ef1:+.2f}  BEDROC={delta_bed:+.3f}  ROC={delta_roc:+.3f}')
    print()
