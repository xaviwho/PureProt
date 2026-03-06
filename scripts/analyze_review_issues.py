#!/usr/bin/env python3
"""
Post-hoc analyses addressing reviewer concerns.

Computes:
1. DUD-E chemical space overlap (Tanimoto) -- why esr1/pparg fail
2. Exhaustiveness sensitivity (e=1 vs e=4 from existing caches)
3. Wilcoxon signed-rank test (regression vs hybrid BEDROC)
4. Brier scores for classification calibration quality
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress rdkit warnings
import warnings
warnings.filterwarnings('ignore', message='.*please use MorganGenerator.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def analyze_dude_chemical_space():
    """Compute Tanimoto overlap between ChEMBL training sets and DUD-E actives."""
    print("=" * 70)
    print("ANALYSIS 1: DUD-E Chemical Space Overlap")
    print("=" * 70)

    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit import DataStructs
    import pandas as pd
    from sklearn.model_selection import train_test_split

    DUDE_MAP = {
        'aa2ar': ('CHEMBL251', 'modeling/data/chembl251_data.csv'),
        'esr1':  ('CHEMBL1862', 'modeling/data/chembl1862_data.csv'),
        'hivpr': ('CHEMBL243', 'chembl243_prepared_data.csv'),
        'pparg': ('CHEMBL4005', 'modeling/data/chembl4005_data.csv'),
        'vgfr2': ('CHEMBL279', 'chembl279_prepared_data.csv'),
    }

    results = {}

    for dude_name, (chembl_id, csv_file) in DUDE_MAP.items():
        print(f"\n--- {dude_name} -> {chembl_id} ---")

        # Load ChEMBL training SMILES
        csv_path = PROJECT_ROOT / csv_file
        df = pd.read_csv(csv_path)
        smiles_col = None
        for col in ['smiles', 'canonical_smiles', 'SMILES']:
            if col in df.columns:
                smiles_col = col
                break
        all_smiles = df[smiles_col].values

        # Replicate 60/20/20 split
        indices = np.arange(len(all_smiles))
        idx_trainval, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
        idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, random_state=42)
        train_smiles = [all_smiles[i] for i in idx_train]

        # Load DUD-E actives
        actives_file = PROJECT_ROOT / 'dude_data' / dude_name / 'actives_final.ism'
        dude_actives = []
        with open(actives_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    dude_actives.append(parts[0])

        print(f"  ChEMBL train: {len(train_smiles)}, DUD-E actives: {len(dude_actives)}")

        # Compute Morgan fingerprints
        def get_fp(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        train_fps = [get_fp(s) for s in train_smiles]
        train_fps = [fp for fp in train_fps if fp is not None]

        dude_fps = [get_fp(s) for s in dude_actives]
        dude_fps_valid = [(i, fp) for i, fp in enumerate(dude_fps) if fp is not None]

        print(f"  Valid FPs: train={len(train_fps)}, dude={len(dude_fps_valid)}")

        # Compute nearest-neighbor Tanimoto for each DUD-E active
        nn_similarities = []
        for idx, dude_fp in dude_fps_valid:
            sims = DataStructs.BulkTanimotoSimilarity(dude_fp, train_fps)
            nn_sim = max(sims)
            nn_similarities.append(nn_sim)

        nn_sims = np.array(nn_similarities)

        # Summary stats
        mean_nn = np.mean(nn_sims)
        median_nn = np.median(nn_sims)
        pct_below_03 = np.mean(nn_sims < 0.3) * 100
        pct_below_04 = np.mean(nn_sims < 0.4) * 100
        pct_above_07 = np.mean(nn_sims >= 0.7) * 100

        print(f"  Nearest-neighbor Tanimoto to ChEMBL training:")
        print(f"    Mean:   {mean_nn:.3f}")
        print(f"    Median: {median_nn:.3f}")
        print(f"    <0.3:   {pct_below_03:.1f}%")
        print(f"    <0.4:   {pct_below_04:.1f}%")
        print(f"    >=0.7:  {pct_above_07:.1f}%")
        print(f"    Min:    {np.min(nn_sims):.3f}")
        print(f"    Max:    {np.max(nn_sims):.3f}")
        print(f"    Q25:    {np.percentile(nn_sims, 25):.3f}")
        print(f"    Q75:    {np.percentile(nn_sims, 75):.3f}")

        results[dude_name] = {
            'chembl_id': chembl_id,
            'n_train': len(train_fps),
            'n_dude_actives': len(dude_fps_valid),
            'nn_tanimoto_mean': float(mean_nn),
            'nn_tanimoto_median': float(median_nn),
            'nn_tanimoto_min': float(np.min(nn_sims)),
            'nn_tanimoto_max': float(np.max(nn_sims)),
            'nn_tanimoto_q25': float(np.percentile(nn_sims, 25)),
            'nn_tanimoto_q75': float(np.percentile(nn_sims, 75)),
            'pct_below_0.3': float(pct_below_03),
            'pct_below_0.4': float(pct_below_04),
            'pct_above_0.7': float(pct_above_07),
        }

    # Print comparison table
    print("\n\nSummary Table:")
    print(f"{'Target':<10} {'ChEMBL':<12} {'N_train':<8} {'N_dude':<8} "
          f"{'NN_mean':<8} {'NN_med':<8} {'<0.3%':<8} {'<0.4%':<8} {'>=0.7%':<8}")
    print("-" * 88)
    for name, r in results.items():
        print(f"{name:<10} {r['chembl_id']:<12} {r['n_train']:<8} {r['n_dude_actives']:<8} "
              f"{r['nn_tanimoto_mean']:<8.3f} {r['nn_tanimoto_median']:<8.3f} "
              f"{r['pct_below_0.3']:<8.1f} {r['pct_below_0.4']:<8.1f} "
              f"{r['pct_above_0.7']:<8.1f}")

    return results


def analyze_exhaustiveness_sensitivity():
    """Compare e=1 vs e=4 docking scores from existing caches."""
    print("\n\n" + "=" * 70)
    print("ANALYSIS 2: Exhaustiveness Sensitivity (e=1 vs e=4)")
    print("=" * 70)

    cache_dir = PROJECT_ROOT / 'docking_cache'
    targets = ['CHEMBL243', 'CHEMBL247', 'CHEMBL279', 'CHEMBL3471', 'CHEMBL2487',
               'CHEMBL251', 'CHEMBL217', 'CHEMBL1862', 'CHEMBL4005', 'CHEMBL240']

    results = {}

    for target in targets:
        for set_name in ['val', 'test']:
            e1_file = cache_dir / f'{target}_{set_name}_vina.json'
            e4_file = cache_dir / f'{target}_{set_name}_vina_e4.json'

            if not e1_file.exists() or not e4_file.exists():
                continue

            with open(e1_file, 'r') as f:
                e1_data = json.load(f)
            with open(e4_file, 'r') as f:
                e4_data = json.load(f)

            e1_scores = np.array(e1_data['scores'])
            e4_scores = np.array(e4_data['scores'])

            if len(e1_scores) != len(e4_scores):
                print(f"  [{target} {set_name}] Size mismatch: e1={len(e1_scores)}, e4={len(e4_scores)}")
                continue

            # Compare only where both have valid (non-zero) scores
            both_valid = (e1_scores != 0.0) & (e4_scores != 0.0)
            n_both = int(both_valid.sum())

            if n_both < 10:
                print(f"  [{target} {set_name}] Too few overlapping valid scores: {n_both}")
                continue

            e1_v = e1_scores[both_valid]
            e4_v = e4_scores[both_valid]

            # Correlation
            pearson_r, pearson_p = stats.pearsonr(e1_v, e4_v)
            spearman_r, spearman_p = stats.spearmanr(e1_v, e4_v)

            # Score differences
            diffs = e4_v - e1_v
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            mae = np.mean(np.abs(diffs))

            # Valid counts
            e1_valid = int(np.sum(e1_scores != 0.0))
            e4_valid = int(np.sum(e4_scores != 0.0))

            key = f"{target}_{set_name}"
            results[key] = {
                'target': target,
                'set': set_name,
                'n_total': len(e1_scores),
                'e1_valid': e1_valid,
                'e4_valid': e4_valid,
                'n_both_valid': n_both,
                'pearson_r': float(pearson_r),
                'spearman_r': float(spearman_r),
                'mean_diff': float(mean_diff),
                'std_diff': float(std_diff),
                'mae': float(mae),
                'e1_mean': float(np.mean(e1_v)),
                'e4_mean': float(np.mean(e4_v)),
            }

    # Summary table
    print(f"\n{'Target':<12} {'Set':<5} {'N':<6} {'e1_valid':<10} {'e4_valid':<10} "
          f"{'Pearson':<8} {'Spearman':<9} {'MAE':<6} {'e1_mean':<9} {'e4_mean':<9}")
    print("-" * 94)

    pearson_vals = []
    spearman_vals = []
    mae_vals = []

    for key, r in sorted(results.items()):
        print(f"{r['target']:<12} {r['set']:<5} {r['n_total']:<6} "
              f"{r['e1_valid']:<10} {r['e4_valid']:<10} "
              f"{r['pearson_r']:<8.3f} {r['spearman_r']:<9.3f} "
              f"{r['mae']:<6.2f} {r['e1_mean']:<9.2f} {r['e4_mean']:<9.2f}")
        pearson_vals.append(r['pearson_r'])
        spearman_vals.append(r['spearman_r'])
        mae_vals.append(r['mae'])

    if pearson_vals:
        print(f"\nMean Pearson r:  {np.mean(pearson_vals):.3f} +/- {np.std(pearson_vals):.3f}")
        print(f"Mean Spearman r: {np.mean(spearman_vals):.3f} +/- {np.std(spearman_vals):.3f}")
        print(f"Mean MAE:        {np.mean(mae_vals):.2f} +/- {np.std(mae_vals):.2f} kcal/mol")

    return results


def analyze_wilcoxon_test():
    """Paired Wilcoxon signed-rank test: regression vs hybrid BEDROC across 10 targets."""
    print("\n\n" + "=" * 70)
    print("ANALYSIS 3: Wilcoxon Signed-Rank Test (Regression vs Hybrid)")
    print("=" * 70)

    results_file = PROJECT_ROOT / 'experiments' / 'paper_results' / 'revised_results.json'
    with open(results_file, 'r') as f:
        data = json.load(f)

    exp2 = data['experiments']['exp2_enrichment_metrics']

    reg_bedrocs = []
    hyb_bedrocs = []
    reg_aucrocs = []
    hyb_aucrocs = []
    targets = []

    for tid, res in exp2.items():
        if not isinstance(res, dict) or res.get('status') != 'SUCCESS':
            continue
        reg_b = res['regression']['bedroc_20']
        hyb_b = res['hybrid']['bedroc_20']
        reg_a = res['regression']['auc_roc']
        hyb_a = res['hybrid']['auc_roc']
        targets.append(tid)
        reg_bedrocs.append(reg_b)
        hyb_bedrocs.append(hyb_b)
        reg_aucrocs.append(reg_a)
        hyb_aucrocs.append(hyb_a)

    reg_bedrocs = np.array(reg_bedrocs)
    hyb_bedrocs = np.array(hyb_bedrocs)
    reg_aucrocs = np.array(reg_aucrocs)
    hyb_aucrocs = np.array(hyb_aucrocs)

    print(f"\nPer-target BEDROC comparison (n={len(targets)}):")
    print(f"{'Target':<12} {'Reg BEDROC':<12} {'Hyb BEDROC':<12} {'Diff':<8}")
    print("-" * 44)
    for i, tid in enumerate(targets):
        diff = hyb_bedrocs[i] - reg_bedrocs[i]
        print(f"{tid:<12} {reg_bedrocs[i]:<12.3f} {hyb_bedrocs[i]:<12.3f} {diff:+.3f}")

    # Wilcoxon test (two-sided)
    # Handle ties (identical values) - Wilcoxon drops them
    diffs = hyb_bedrocs - reg_bedrocs
    n_nonzero = np.sum(diffs != 0)
    print(f"\nBEDROC differences: {n_nonzero}/{len(diffs)} non-zero")

    if n_nonzero >= 2:
        stat_b, p_b = stats.wilcoxon(reg_bedrocs, hyb_bedrocs, alternative='two-sided')
        print(f"Wilcoxon BEDROC: statistic={stat_b:.3f}, p={p_b:.4f}")
    else:
        p_b = 1.0
        print(f"Wilcoxon BEDROC: insufficient non-tied pairs (n={n_nonzero})")

    # Also test AUC-ROC
    diffs_auc = hyb_aucrocs - reg_aucrocs
    n_nonzero_auc = np.sum(diffs_auc != 0)
    print(f"\nAUC-ROC differences: {n_nonzero_auc}/{len(diffs_auc)} non-zero")

    if n_nonzero_auc >= 2:
        stat_a, p_a = stats.wilcoxon(reg_aucrocs, hyb_aucrocs, alternative='two-sided')
        print(f"Wilcoxon AUC-ROC: statistic={stat_a:.3f}, p={p_a:.4f}")
    else:
        p_a = 1.0
        print(f"Wilcoxon AUC-ROC: insufficient non-tied pairs (n={n_nonzero_auc})")

    # Also compare classification vs regression
    clf_bedrocs = []
    for tid, res in exp2.items():
        if not isinstance(res, dict) or res.get('status') != 'SUCCESS':
            continue
        clf_bedrocs.append(res['classification']['bedroc_20'])
    clf_bedrocs = np.array(clf_bedrocs)

    diffs_clf = clf_bedrocs - reg_bedrocs
    n_nonzero_clf = np.sum(diffs_clf != 0)
    print(f"\nClassification vs Regression BEDROC: {n_nonzero_clf}/{len(diffs_clf)} non-zero diffs")
    if n_nonzero_clf >= 2:
        stat_c, p_c = stats.wilcoxon(reg_bedrocs, clf_bedrocs, alternative='two-sided')
        print(f"Wilcoxon Clf vs Reg: statistic={stat_c:.3f}, p={p_c:.4f}")

    print(f"\nMean BEDROC - Regression: {np.mean(reg_bedrocs):.3f} +/- {np.std(reg_bedrocs):.3f}")
    print(f"Mean BEDROC - Hybrid:     {np.mean(hyb_bedrocs):.3f} +/- {np.std(hyb_bedrocs):.3f}")
    print(f"Mean BEDROC - Clf:        {np.mean(clf_bedrocs):.3f} +/- {np.std(clf_bedrocs):.3f}")

    return {
        'n_targets': len(targets),
        'reg_bedroc_mean': float(np.mean(reg_bedrocs)),
        'hyb_bedroc_mean': float(np.mean(hyb_bedrocs)),
        'clf_bedroc_mean': float(np.mean(clf_bedrocs)),
        'wilcoxon_bedroc_p': float(p_b),
        'wilcoxon_aucroc_p': float(p_a) if n_nonzero_auc >= 2 else None,
        'targets': targets,
        'reg_bedrocs': reg_bedrocs.tolist(),
        'hyb_bedrocs': hyb_bedrocs.tolist(),
    }


def analyze_brier_scores():
    """Compute Brier scores for classification ensemble calibration quality."""
    print("\n\n" + "=" * 70)
    print("ANALYSIS 4: Classification Calibration (Brier Scores)")
    print("=" * 70)

    from pureprot.ai_model import ConsensusAIModel
    from sklearn.metrics import brier_score_loss

    CHEMBL_DATA_FILES = {
        'CHEMBL243':  'chembl243_prepared_data.csv',
        'CHEMBL247':  'chembl247_data.csv',
        'CHEMBL279':  'chembl279_prepared_data.csv',
        'CHEMBL3471': 'chembl3471_data.csv',
        'CHEMBL2487': 'modeling/data/chembl_2487_data.csv',
        'CHEMBL251':  'modeling/data/chembl251_data.csv',
        'CHEMBL217':  'modeling/data/chembl217_data.csv',
        'CHEMBL1862': 'modeling/data/chembl1862_data.csv',
        'CHEMBL4005': 'modeling/data/chembl4005_data.csv',
        'CHEMBL240':  'modeling/data/chembl240_data.csv',
    }

    results = {}

    for target_id, csv_file in CHEMBL_DATA_FILES.items():
        print(f"\n  [{target_id}] Training model for Brier score ...", flush=True)
        data_path = str(PROJECT_ROOT / csv_file)
        if not os.path.exists(data_path):
            print(f"    Data file not found: {data_path}")
            continue

        try:
            model = ConsensusAIModel(model_type='both')
            model.train(data_path=data_path, activity_threshold=6.0, activity_mode='threshold')

            vd = model._val_data
            X_val = vd['X_val_scaled']
            X_test = vd['X_test_scaled']
            y_val = vd['y_val']
            y_test = vd['y_test']

            y_val_bin = model._make_binary_labels(y_val, threshold=6.0, mode='threshold')
            y_test_bin = model._make_binary_labels(y_test, threshold=6.0, mode='threshold')

            if y_test_bin is None:
                print(f"    Skipping: no binary labels")
                continue

            # Get consensus classification probabilities
            clf_probs_val = model.predict_classification_consensus(X_val)
            clf_probs_test = model.predict_classification_consensus(X_test)

            # Individual classifier probabilities
            individual_brier = {}
            for clf_name, clf in model.classifiers.items():
                probs_test = clf.predict_proba(X_test)[:, 1]
                brier = brier_score_loss(y_test_bin, probs_test)
                individual_brier[clf_name] = float(brier)

            # Consensus Brier
            brier_val = brier_score_loss(y_val_bin, clf_probs_val)
            brier_test = brier_score_loss(y_test_bin, clf_probs_test)

            # Prevalence baseline (always predict class proportion)
            prevalence = y_test_bin.mean()
            brier_baseline = brier_score_loss(y_test_bin, np.full_like(clf_probs_test, prevalence))

            # Brier skill score (BSS = 1 - Brier/Brier_ref)
            bss = 1.0 - brier_test / brier_baseline if brier_baseline > 0 else 0.0

            results[target_id] = {
                'brier_val': float(brier_val),
                'brier_test': float(brier_test),
                'brier_baseline': float(brier_baseline),
                'brier_skill_score': float(bss),
                'individual_brier': individual_brier,
                'prevalence': float(prevalence),
            }

            print(f"    Brier(test)={brier_test:.4f}, baseline={brier_baseline:.4f}, "
                  f"BSS={bss:.3f}, prevalence={prevalence:.2f}")

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n{'Target':<12} {'Brier':<8} {'Baseline':<10} {'BSS':<8} {'Prevalence':<12} "
          f"{'SVC':<8} {'RF':<8} {'GB':<8}")
    print("-" * 78)
    brier_vals = []
    bss_vals = []
    for tid, r in results.items():
        ind = r['individual_brier']
        print(f"{tid:<12} {r['brier_test']:<8.4f} {r['brier_baseline']:<10.4f} "
              f"{r['brier_skill_score']:<8.3f} {r['prevalence']:<12.2f} "
              f"{ind.get('svc', 0):<8.4f} {ind.get('rf_clf', 0):<8.4f} "
              f"{ind.get('gb_clf', 0):<8.4f}")
        brier_vals.append(r['brier_test'])
        bss_vals.append(r['brier_skill_score'])

    if brier_vals:
        print(f"\nMean Brier score: {np.mean(brier_vals):.4f} +/- {np.std(brier_vals):.4f}")
        print(f"Mean BSS:         {np.mean(bss_vals):.3f} +/- {np.std(bss_vals):.3f}")

    return results


def main():
    all_results = {}

    # Analysis 1: DUD-E chemical space overlap
    all_results['dude_overlap'] = analyze_dude_chemical_space()

    # Analysis 2: Exhaustiveness sensitivity
    all_results['exhaustiveness'] = analyze_exhaustiveness_sensitivity()

    # Analysis 3: Wilcoxon test
    all_results['wilcoxon'] = analyze_wilcoxon_test()

    # Analysis 4: Brier scores
    all_results['brier'] = analyze_brier_scores()

    # Save results
    output_path = PROJECT_ROOT / 'experiments' / 'paper_results' / 'review_analyses.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll analysis results saved to: {output_path}")


if __name__ == '__main__':
    main()
