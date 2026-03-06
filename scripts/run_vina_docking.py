#!/usr/bin/env python3
"""
Batch Vina Docking Script for PureProtX

Docks all val/test molecules for each target using AutoDock Vina 1.2.7.
Results are cached to docking_cache/ for use by run_experiments.py.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VINA_EXE = str(PROJECT_ROOT / 'tools' / 'vina.exe')
PREPARED_DIR = str(PROJECT_ROOT / 'structures' / 'prepared')
CACHE_DIR = PROJECT_ROOT / 'docking_cache'
CACHE_DIR.mkdir(exist_ok=True)

# Target docking configurations
TARGET_CONFIG = {
    'CHEMBL243':  {'center': (10.8, 23.1, 3.9),    'box': (22,22,22)},
    'CHEMBL247':  {'center': (10.9, 13.7, 17.3),   'box': (22,22,22)},
    'CHEMBL279':  {'center': (-25.0, -1.1, -10.5),  'box': (24,24,24)},
    'CHEMBL3471': {'center': (44.2, 14.9, 31.4),   'box': (22,22,22)},
    'CHEMBL2487': {'center': (22.0, 23.9, 0.3),    'box': (22,22,22)},
    'CHEMBL251':  {'center': (-0.4, 8.5, 17.1),    'box': (22,22,22)},
    'CHEMBL217':  {'center': (9.9, 5.8, -9.6),     'box': (22,22,22)},
    'CHEMBL1862': {'center': (31.6, -1.6, 25.6),   'box': (22,22,22)},
    'CHEMBL4005': {'center': (54.7, -21.0, 30.5),  'box': (22,22,22)},
    'CHEMBL240':  {'center': (73.1, 73.2, 77.2),   'box': (24,24,24)},
}

# Data file paths (relative to project root)
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


def prepare_ligand_pdbqt(smiles):
    """Convert SMILES to PDBQT string using meeko."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result < 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result < 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return pdbqt_string
    except Exception:
        pass

    return None


def dock_single(args):
    """Dock a single molecule. Returns (index, score)."""
    idx, smiles, receptor_pdbqt, center, box, vina_path, tmp_dir = args

    pdbqt_str = prepare_ligand_pdbqt(smiles)
    if pdbqt_str is None:
        return (idx, 0.0)

    # Use unique temp file names based on index
    lig_path = os.path.join(tmp_dir, f'lig_{idx}.pdbqt')
    out_path = os.path.join(tmp_dir, f'out_{idx}.pdbqt')

    try:
        with open(lig_path, 'w') as f:
            f.write(pdbqt_str)

        cmd = [
            vina_path,
            '--receptor', receptor_pdbqt,
            '--ligand', lig_path,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(box[0]),
            '--size_y', str(box[1]),
            '--size_z', str(box[2]),
            '--exhaustiveness', '4',
            '--num_modes', '1',
            '--cpu', '1',
            '--out', out_path,
            '--verbosity', '0',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        score = 0.0
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, 'r') as f:
                for line in f:
                    if 'REMARK VINA RESULT' in line:
                        score = float(line.split()[3])
                        break

        return (idx, score)

    except Exception:
        return (idx, 0.0)

    finally:
        for p in [lig_path, out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


def dock_target_set(target_id, smiles_list, set_name, max_workers=6):
    """Dock all molecules in a set (val or test) for a target."""
    cache_file = CACHE_DIR / f'{target_id}_{set_name}_vina_e4.json'

    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        if len(cached['scores']) == len(smiles_list):
            valid = sum(1 for s in cached['scores'] if s != 0.0)
            print(f"  [CACHE] {target_id} {set_name}: {valid}/{len(smiles_list)} valid scores")
            return np.array(cached['scores'])

    cfg = TARGET_CONFIG[target_id]
    receptor = os.path.join(PREPARED_DIR, f'{target_id}_receptor.pdbqt')
    center = cfg['center']
    box = cfg['box']

    n = len(smiles_list)
    print(f"  Docking {target_id} {set_name}: {n} molecules...", flush=True)

    # Create temp directory for this batch
    tmp_dir = str(CACHE_DIR / f'tmp_{target_id}_{set_name}')
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare work items
    work_items = [
        (i, smi, receptor, center, box, VINA_EXE, tmp_dir)
        for i, smi in enumerate(smiles_list)
    ]

    scores = np.zeros(n)
    completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(dock_single, item): item[0]
            for item in work_items
        }

        for future in as_completed(futures):
            idx, score = future.result()
            scores[idx] = score
            completed += 1

            if completed % 25 == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n - completed) / rate if rate > 0 else 0
                print(f"    [{target_id} {set_name}] {completed}/{n} "
                      f"({rate:.1f} mol/s, ETA {eta:.0f}s)", flush=True)

    # Cleanup temp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # Save cache
    valid = int(np.sum(scores != 0.0))
    mean_score = float(np.mean(scores[scores != 0.0])) if valid > 0 else 0.0
    cache_data = {
        'target_id': target_id,
        'set_name': set_name,
        'n_molecules': n,
        'n_valid': valid,
        'mean_score': mean_score,
        'timestamp': time.time(),
        'scores': scores.tolist(),
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    elapsed = time.time() - t0
    print(f"  [{target_id} {set_name}] Done: {valid}/{n} valid, "
          f"mean={mean_score:.2f} kcal/mol, time={elapsed:.0f}s", flush=True)

    return scores


def load_and_split_data(target_id):
    """Load data and apply 60/20/20 split matching run_experiments.py."""
    from sklearn.model_selection import train_test_split

    csv_file = CHEMBL_DATA_FILES[target_id]
    csv_path = PROJECT_ROOT / csv_file
    if not csv_path.exists():
        print(f"  [ERROR] Data file not found: {csv_path}")
        return None, None, None

    df = pd.read_csv(csv_path)

    # Support same column name variants as pureprot/ai_model.py
    smiles_col = None
    for col_name in ['smiles', 'canonical_smiles', 'SMILES']:
        if col_name in df.columns:
            smiles_col = col_name
            break

    pic50_col = None
    for col_name in ['pic50', 'pIC50', 'pchembl_value']:
        if col_name in df.columns:
            pic50_col = col_name
            break

    if smiles_col is None or pic50_col is None:
        print(f"  [ERROR] Missing columns in {csv_path}. Found: {list(df.columns)}")
        return None, None, None

    smiles = df[smiles_col].values
    y = df[pic50_col].values

    # Stage 1: 80/20 split
    idx_trainval, idx_test = train_test_split(
        np.arange(len(smiles)), test_size=0.2, random_state=42
    )
    # Stage 2: 75/25 on trainval
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=42
    )

    return (
        smiles[idx_train].tolist(),
        smiles[idx_val].tolist(),
        smiles[idx_test].tolist(),
    )


def main():
    print("=" * 70)
    print("PureProtX Vina Batch Docking")
    print(f"Vina: {VINA_EXE}")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 70)

    max_workers = 10
    total_t0 = time.time()

    # Sort targets by dataset size (smallest first) for faster partial results
    target_sizes = {}
    for target_id in TARGET_CONFIG:
        train_smi, val_smi, test_smi = load_and_split_data(target_id)
        if val_smi is not None:
            target_sizes[target_id] = (len(val_smi) + len(test_smi), train_smi, val_smi, test_smi)

    sorted_targets = sorted(target_sizes.keys(), key=lambda t: target_sizes[t][0])
    print(f"\nTarget order (smallest first):", flush=True)
    for t in sorted_targets:
        n = target_sizes[t][0]
        print(f"  {t}: {n} molecules (val+test)", flush=True)

    for target_id in sorted_targets:
        _, train_smi, val_smi, test_smi = target_sizes[target_id]
        print(f"\n--- {target_id} ---", flush=True)
        print(f"  Train: {len(train_smi)}, Val: {len(val_smi)}, Test: {len(test_smi)}", flush=True)

        # Dock val and test sets
        dock_target_set(target_id, val_smi, 'val', max_workers=max_workers)
        dock_target_set(target_id, test_smi, 'test', max_workers=max_workers)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"Total docking time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
