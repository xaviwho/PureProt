#!/usr/bin/env python3
"""Quick analysis of Vina docking scores vs regression for completed targets."""
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
}

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

    fAI_val = normalize_ai_scores(model.predict_consensus(X_val))
    fAI_test = normalize_ai_scores(model.predict_consensus(X_test))

    # Load raw Vina scores
    vina = {}
    for sn in ['val', 'test']:
        with open(f'docking_cache/{target_id}_{sn}_vina.json') as f:
            vina[sn] = np.array(json.load(f)['scores'])

    # Failure analysis
    for sn, labels in [('val', y_val_bin), ('test', y_test_bin)]:
        raw = vina[sn]
        failed = (raw == 0.0)
        n_act = int(np.sum(labels == 1))
        n_act_fail = int(np.sum((labels == 1) & failed))
        n_inact = int(np.sum(labels == 0))
        n_inact_fail = int(np.sum((labels == 0) & failed))
        pct_act = 100 * n_act_fail / max(1, n_act)
        pct_inact = 100 * n_inact_fail / max(1, n_inact)
        print(f'  {sn}: {int(np.sum(failed))}/{len(raw)} fail | '
              f'act_fail={n_act_fail}/{n_act} ({pct_act:.0f}%) | '
              f'inact_fail={n_inact_fail}/{n_inact} ({pct_inact:.0f}%)')

    # Median imputation for failed dockings
    for sn in ['val', 'test']:
        raw = vina[sn]
        valid_scores = raw[raw != 0.0]
        med = float(np.median(valid_scores))
        imputed = raw.copy()
        imputed[raw == 0.0] = med
        vina[sn + '_imp'] = imputed

    fdock_val = normalize_docking_scores(vina['val_imp'])
    fdock_test = normalize_docking_scores(vina['test_imp'])

    # Correlation
    corr = np.corrcoef(fAI_test, fdock_test)[0, 1]
    print(f'  Corr(fAI, fDock): {corr:.3f}')

    # Baseline
    reg_m = evaluate_ranking(fAI_test, y_test_bin, 'reg')
    print(f'  Regression: EF1={reg_m["ef1"]:.2f} BEDROC={reg_m["bedroc_20"]:.3f} ROC={reg_m["auc_roc"]:.3f}')

    # Alpha sweep
    for a in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
        hybrid = a * fAI_test + (1 - a) * fdock_test
        m = evaluate_ranking(hybrid, y_test_bin, f'a={a}')
        db = m['bedroc_20'] - reg_m['bedroc_20']
        print(f'  a={a:.2f}: EF1={m["ef1"]:.2f} BEDROC={m["bedroc_20"]:.3f}({db:+.3f}) ROC={m["auc_roc"]:.3f}')

    # Optimized alpha on val (with imputation)
    for metric in ['ef1', 'bedroc']:
        best_a, best_v = optimize_alpha(fAI_val, fdock_val, y_val_bin, metric=metric)
        hybrid = best_a * fAI_test + (1 - best_a) * fdock_test
        m = evaluate_ranking(hybrid, y_test_bin, f'opt-{metric}')
        db = m['bedroc_20'] - reg_m['bedroc_20']
        print(f'  opt-{metric}(a={best_a:.2f}): EF1={m["ef1"]:.2f} BEDROC={m["bedroc_20"]:.3f}({db:+.3f}) ROC={m["auc_roc"]:.3f}')

    print()
