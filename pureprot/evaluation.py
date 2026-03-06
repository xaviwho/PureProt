"""
Evaluation Module for PureProtX

Implements enrichment metrics (EF, BEDROC, AUC) and alpha optimization
for hybrid AI+docking consensus scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_enrichment_factor(scores: np.ndarray, labels: np.ndarray,
                               percent: float = 1.0) -> float:
    """
    Compute Enrichment Factor at a given percentage.

    EF_x% = (actives_in_top_x% / N_top_x%) / (total_actives / N_total)

    Args:
        scores: Predicted scores (higher = more likely active)
        labels: Binary labels (1=active, 0=inactive)
        percent: Top percentage to evaluate (e.g., 1.0, 5.0, 10.0)

    Returns:
        Enrichment factor value
    """
    n_total = len(scores)
    n_actives_total = labels.sum()

    if n_actives_total == 0 or n_total == 0:
        return 0.0

    n_top = max(1, int(np.ceil(n_total * percent / 100.0)))

    # Rank by descending score
    ranked_idx = np.argsort(-scores)
    top_labels = labels[ranked_idx[:n_top]]
    n_actives_top = top_labels.sum()

    # EF = (actives_found / n_top) / (total_actives / n_total)
    random_rate = n_actives_total / n_total
    if random_rate == 0:
        return 0.0

    ef = (n_actives_top / n_top) / random_rate
    return float(ef)


def compute_bedroc(scores: np.ndarray, labels: np.ndarray,
                   alpha: float = 20.0) -> float:
    """
    Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC is a metric that emphasizes early enrichment more heavily than AUC-ROC.
    Alpha controls the exponential weighting (default 20 = top ~8% of ranked list).

    Args:
        scores: Predicted scores (higher = more likely active)
        labels: Binary labels (1=active, 0=inactive)
        alpha: Exponential weighting parameter (default 20.0)

    Returns:
        BEDROC score between 0 and 1
    """
    n = len(scores)
    n_actives = int(labels.sum())

    if n_actives == 0 or n == 0:
        return 0.0

    # Sort by descending score
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # Get ranks of actives (1-indexed, normalized to [0,1])
    active_ranks = np.where(labels_sorted == 1)[0]
    # Normalize to fractional ranks (0 to 1)
    ri = (active_ranks + 1) / n

    # Compute BEDROC
    s = np.sum(np.exp(-alpha * ri))

    # Random sum
    ra = n_actives / n
    rand_sum = (ra * (1 - np.exp(-alpha))) / (np.exp(alpha / n) - 1)

    # Perfect sum
    perfect_sum = (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha / n))

    if perfect_sum - rand_sum == 0:
        return 0.0

    bedroc = (s - rand_sum) / (perfect_sum - rand_sum)
    return float(np.clip(bedroc, 0.0, 1.0))


def optimize_alpha(fAI: np.ndarray, fdock: np.ndarray, labels: np.ndarray,
                   metric: str = 'ef1', alpha_range: np.ndarray = None) -> Tuple[float, float]:
    """
    Optimize the alpha weighting parameter on a validation set.

    consensus_score = alpha * fAI + (1 - alpha) * fdock

    Args:
        fAI: Normalized AI scores (0-1 scale)
        fdock: Normalized docking scores (0-1 scale)
        labels: Binary labels (1=active, 0=inactive)
        metric: Optimization target ('ef1', 'ef5', 'ef10', 'bedroc', 'auc_roc')
        alpha_range: Array of alpha candidates to search

    Returns:
        Tuple of (best_alpha, best_metric_value)
    """
    if alpha_range is None:
        alpha_range = np.arange(0.0, 1.05, 0.05)

    best_alpha = 0.5
    best_score = -np.inf

    for alpha in alpha_range:
        scores = alpha * fAI + (1 - alpha) * fdock

        if metric == 'ef1':
            score = compute_enrichment_factor(scores, labels, percent=1.0)
        elif metric == 'ef5':
            score = compute_enrichment_factor(scores, labels, percent=5.0)
        elif metric == 'ef10':
            score = compute_enrichment_factor(scores, labels, percent=10.0)
        elif metric == 'bedroc':
            score = compute_bedroc(scores, labels, alpha=20.0)
        elif metric == 'auc_roc':
            if len(np.unique(labels)) < 2:
                continue
            score = roc_auc_score(labels, scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return float(best_alpha), float(best_score)


def evaluate_ranking(scores: np.ndarray, labels: np.ndarray,
                     method_name: str = '') -> Dict[str, float]:
    """
    Compute a full suite of ranking metrics.

    Args:
        scores: Predicted scores (higher = more likely active)
        labels: Binary labels (1=active, 0=inactive)
        method_name: Label for this scoring method

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        'method': method_name,
        'ef1': compute_enrichment_factor(scores, labels, 1.0),
        'ef5': compute_enrichment_factor(scores, labels, 5.0),
        'ef10': compute_enrichment_factor(scores, labels, 10.0),
        'bedroc_20': compute_bedroc(scores, labels, alpha=20.0),
    }

    if len(np.unique(labels)) > 1:
        metrics['auc_roc'] = float(roc_auc_score(labels, scores))
        metrics['auc_pr'] = float(average_precision_score(labels, scores))

    return metrics


def compare_methods(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a comparison table across multiple scoring methods.

    Args:
        results_dict: {method_name: metrics_dict}

    Returns:
        DataFrame with methods as rows and metrics as columns
    """
    rows = []
    for method_name, metrics in results_dict.items():
        row = {'method': method_name}
        row.update({k: v for k, v in metrics.items() if k != 'method'})
        rows.append(row)

    df = pd.DataFrame(rows)
    if 'method' in df.columns:
        df = df.set_index('method')

    return df


def normalize_ai_scores(pic50: np.ndarray, low: float = 4.0,
                         high: float = 10.0) -> np.ndarray:
    """Normalize pIC50 values to [0,1] via per-target z-score.

    Symmetric with normalize_docking_scores so both components
    have comparable variance for unbiased alpha optimization.
    """
    if len(pic50) < 2:
        return np.zeros_like(pic50, dtype=float)
    mu = np.mean(pic50)
    sigma = np.std(pic50)
    if sigma < 1e-10:
        return np.full_like(pic50, 0.5, dtype=float)
    z = (pic50 - mu) / sigma  # higher pIC50 = better = higher score
    return np.clip((z + 3.0) / 6.0, 0.0, 1.0)


def normalize_docking_scores(energies: np.ndarray,
                              max_neg: float = 15.0) -> np.ndarray:
    """Normalize docking energies to [0,1] via per-target z-score.

    More negative energy = better binding = higher normalized score.
    Failed dockings (already imputed with median by load_vina_scores)
    are handled naturally by z-scoring.
    """
    if len(energies) < 2:
        return np.zeros_like(energies, dtype=float)
    mu = np.mean(energies)
    sigma = np.std(energies)
    if sigma < 1e-10:
        return np.full_like(energies, 0.5, dtype=float)
    # Negate so more negative energy maps to higher score
    z = -(energies - mu) / sigma
    # Map +/-3 sigma to [0, 1]
    return np.clip((z + 3.0) / 6.0, 0.0, 1.0)


def leave_one_target_out_alpha(target_data: Dict[str, Dict[str, np.ndarray]],
                                metric: str = 'ef1') -> Dict[str, Any]:
    """
    Leave-one-target-out cross-validation for alpha optimization.

    For each target t, optimize alpha on validation sets of all OTHER targets,
    then evaluate on target t's test set.

    Args:
        target_data: Dict of target_name -> {
            'fAI_val': normalized AI scores (val),
            'fdock_val': normalized docking scores (val),
            'labels_val': binary labels (val),
            'fAI_test': normalized AI scores (test),
            'fdock_test': normalized docking scores (test),
            'labels_test': binary labels (test),
        }
        metric: Metric to optimize ('ef1', 'ef5', 'bedroc', 'auc_roc')

    Returns:
        Dict with per-target results and mean optimal alpha
    """
    target_names = list(target_data.keys())
    results = {}
    alphas = []

    for held_out in target_names:
        # Pool validation data from all OTHER targets
        fAI_pool = []
        fdock_pool = []
        labels_pool = []

        for t in target_names:
            if t == held_out:
                continue
            fAI_pool.append(target_data[t]['fAI_val'])
            fdock_pool.append(target_data[t]['fdock_val'])
            labels_pool.append(target_data[t]['labels_val'])

        fAI_pool = np.concatenate(fAI_pool)
        fdock_pool = np.concatenate(fdock_pool)
        labels_pool = np.concatenate(labels_pool)

        # Optimize alpha on pooled validation data
        best_alpha, val_score = optimize_alpha(
            fAI_pool, fdock_pool, labels_pool, metric=metric
        )

        # Evaluate on held-out target's test set
        test = target_data[held_out]
        test_scores = best_alpha * test['fAI_test'] + (1 - best_alpha) * test['fdock_test']
        test_metrics = evaluate_ranking(test_scores, test['labels_test'],
                                         method_name=f'hybrid_alpha={best_alpha:.2f}')

        results[held_out] = {
            'alpha': best_alpha,
            'val_score': val_score,
            'test_metrics': test_metrics
        }
        alphas.append(best_alpha)

    mean_alpha = float(np.mean(alphas))
    std_alpha = float(np.std(alphas))

    return {
        'per_target': results,
        'mean_alpha': mean_alpha,
        'std_alpha': std_alpha,
        'recommended_alpha': mean_alpha,
        'metric': metric
    }
