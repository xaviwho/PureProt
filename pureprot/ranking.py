"""
Ranking Module for PureProtX

Implements deterministic tie-breaking and score normalization
for reproducible molecule ranking.
"""

import pandas as pd
import numpy as np
from typing import Optional


def canonicalize_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES for deterministic ordering."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except ImportError:
        return smiles  # Fallback: use string as-is if rdkit unavailable


def rank_with_tiebreaking(df: pd.DataFrame,
                           score_col: str = 'consensus_score',
                           docking_col: Optional[str] = 'docking_energy',
                           smiles_col: str = 'smiles') -> pd.DataFrame:
    """
    Rank molecules with deterministic tie-breaking.

    Priority:
      1. Descending consensus score (primary)
      2. Ascending docking energy — more negative = better (secondary)
      3. Lexicographic canonical SMILES (tertiary, fully deterministic)

    Args:
        df: DataFrame with scores and SMILES
        score_col: Column name for primary score
        docking_col: Column name for docking energy (optional)
        smiles_col: Column name for SMILES

    Returns:
        Sorted DataFrame with 'rank' column added
    """
    result = df.copy()

    # Ensure canonical SMILES for deterministic ordering
    if smiles_col in result.columns:
        result['_canonical_smiles'] = result[smiles_col].apply(canonicalize_smiles)
    else:
        result['_canonical_smiles'] = ''

    # Build sort columns
    sort_cols = [score_col]
    ascending = [False]  # Higher score = better

    if docking_col and docking_col in result.columns:
        sort_cols.append(docking_col)
        ascending.append(True)  # More negative = better

    sort_cols.append('_canonical_smiles')
    ascending.append(True)  # Lexicographic

    result = result.sort_values(
        by=sort_cols,
        ascending=ascending
    ).reset_index(drop=True)

    result['rank'] = range(1, len(result) + 1)
    result = result.drop(columns=['_canonical_smiles'])

    return result
