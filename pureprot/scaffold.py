"""
Scaffold Diversity Analysis Module for PureProtX

Implements Murcko scaffold extraction, novelty metrics, and
Tanimoto similarity analysis for evaluating hit diversity.
"""

import warnings
warnings.filterwarnings('ignore', message='.*please use MorganGenerator.*')
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str) -> Optional[str]:
    """
    Extract Murcko scaffold from a SMILES string.

    Args:
        smiles: Canonical SMILES

    Returns:
        SMILES of the Murcko scaffold, or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


def get_generic_scaffold(smiles: str) -> Optional[str]:
    """
    Extract generic (framework) Murcko scaffold (all atoms -> carbon, all bonds -> single).

    Args:
        smiles: Canonical SMILES

    Returns:
        Generic scaffold SMILES, or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


def compute_morgan_fp(smiles: str, radius: int = 2,
                      n_bits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def nearest_train_similarity(hit_fp: DataStructs.ExplicitBitVect,
                              train_fps: List[DataStructs.ExplicitBitVect]) -> float:
    """
    Compute maximum Tanimoto similarity between a hit and all training compounds.

    Args:
        hit_fp: Morgan fingerprint of hit compound
        train_fps: List of Morgan fingerprints for training compounds

    Returns:
        Maximum Tanimoto similarity (0-1)
    """
    if not train_fps:
        return 0.0
    similarities = DataStructs.BulkTanimotoSimilarity(hit_fp, train_fps)
    return float(max(similarities))


def analyze_scaffold_diversity(hit_smiles: List[str],
                                train_smiles: List[str],
                                method_name: str = '') -> Dict:
    """
    Comprehensive scaffold diversity analysis for a set of hits.

    Args:
        hit_smiles: SMILES of predicted hits (top-N from a ranking method)
        train_smiles: SMILES of training set compounds
        method_name: Label for this method

    Returns:
        Dictionary of diversity metrics
    """
    # Extract scaffolds
    hit_scaffolds = [get_scaffold(s) for s in hit_smiles]
    hit_scaffolds_valid = [s for s in hit_scaffolds if s is not None]

    train_scaffolds = set()
    for s in train_smiles:
        sc = get_scaffold(s)
        if sc is not None:
            train_scaffolds.add(sc)

    # Unique scaffold count
    unique_hit_scaffolds = set(hit_scaffolds_valid)
    n_unique = len(unique_hit_scaffolds)

    # Novel scaffold fraction: scaffolds NOT in training set
    novel_scaffolds = unique_hit_scaffolds - train_scaffolds
    novel_fraction = len(novel_scaffolds) / max(1, n_unique)

    # Scaffold recovery rate: fraction of known active scaffolds recovered
    recovered = unique_hit_scaffolds & train_scaffolds
    recovery_rate = len(recovered) / max(1, len(train_scaffolds))

    # Compute Tanimoto similarity to training set
    train_fps = []
    for s in train_smiles:
        fp = compute_morgan_fp(s)
        if fp is not None:
            train_fps.append(fp)

    tanimoto_similarities = []
    for s in hit_smiles:
        fp = compute_morgan_fp(s)
        if fp is not None and train_fps:
            sim = nearest_train_similarity(fp, train_fps)
            tanimoto_similarities.append(sim)

    tanimoto_arr = np.array(tanimoto_similarities) if tanimoto_similarities else np.array([0.0])

    return {
        'method': method_name,
        'n_hits': len(hit_smiles),
        'n_unique_scaffolds': n_unique,
        'scaffold_diversity': n_unique / max(1, len(hit_smiles)),
        'n_novel_scaffolds': len(novel_scaffolds),
        'novel_scaffold_fraction': novel_fraction,
        'scaffold_recovery_rate': recovery_rate,
        'tanimoto_mean': float(tanimoto_arr.mean()),
        'tanimoto_median': float(np.median(tanimoto_arr)),
        'tanimoto_std': float(tanimoto_arr.std()),
        'tanimoto_min': float(tanimoto_arr.min()),
        'tanimoto_max': float(tanimoto_arr.max()),
    }


def compare_scaffold_diversity(ranked_results: Dict[str, pd.DataFrame],
                                train_smiles: List[str],
                                top_percents: List[float] = None) -> pd.DataFrame:
    """
    Compare scaffold diversity across multiple ranking methods at different cutoffs.

    Args:
        ranked_results: {method_name: DataFrame with 'smiles' sorted by score desc}
        train_smiles: SMILES of training set
        top_percents: Cutoff percentages (default: [1, 5, 10])

    Returns:
        DataFrame with diversity metrics per method per cutoff
    """
    if top_percents is None:
        top_percents = [1.0, 5.0, 10.0]

    rows = []
    for method_name, df in ranked_results.items():
        all_smiles = df['smiles'].tolist()
        n_total = len(all_smiles)

        for pct in top_percents:
            n_top = max(1, int(np.ceil(n_total * pct / 100.0)))
            top_smiles = all_smiles[:n_top]

            metrics = analyze_scaffold_diversity(top_smiles, train_smiles, method_name)
            metrics['cutoff_pct'] = pct
            metrics['n_top'] = n_top
            rows.append(metrics)

    return pd.DataFrame(rows)
