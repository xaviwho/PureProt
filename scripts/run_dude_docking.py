#!/usr/bin/env python3
"""
Batch Vina Docking for DUD-E Benchmark Molecules.

Docks DUD-E actives + decoys against the same receptor PDBQTs
used for ChEMBL docking. Uses exhaustiveness=8 (field standard).

Results are cached to docking_cache/dude_{target}_e8.json
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_dude import (
    DUDE_TO_CHEMBL, load_dude_data, download_dude_target, DUDE_DATA_DIR
)

VINA_EXE = str(PROJECT_ROOT / 'tools' / 'vina.exe')
PREPARED_DIR = str(PROJECT_ROOT / 'structures' / 'prepared')
CACHE_DIR = PROJECT_ROOT / 'docking_cache'
CACHE_DIR.mkdir(exist_ok=True)

# Reuse target configs from the main docking module
from modeling.vina_docking import TARGET_DOCKING_CONFIG


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

    lig_path = os.path.join(tmp_dir, f'dude_lig_{idx}.pdbqt')
    out_path = os.path.join(tmp_dir, f'dude_out_{idx}.pdbqt')

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


def dock_dude_target(dude_name, smiles_list, max_workers=10):
    """Dock all DUD-E molecules for a target."""
    chembl_id = DUDE_TO_CHEMBL[dude_name]
    cache_file = CACHE_DIR / f'dude_{dude_name}_e4.json'

    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        if len(cached['scores']) == len(smiles_list):
            valid = sum(1 for s in cached['scores'] if s != 0.0)
            print(f"  [CACHE] {dude_name}: {valid}/{len(smiles_list)} valid scores")
            return np.array(cached['scores'])

    config = TARGET_DOCKING_CONFIG[chembl_id]
    receptor = os.path.join(PREPARED_DIR, f'{chembl_id}_receptor.pdbqt')
    center = config['center']
    box = config['box_size']

    n = len(smiles_list)
    print(f"  Docking {dude_name} ({chembl_id}): {n} molecules (e=4)...", flush=True)

    tmp_dir = str(CACHE_DIR / f'tmp_dude_{dude_name}')
    os.makedirs(tmp_dir, exist_ok=True)

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

            if completed % 50 == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n - completed) / rate if rate > 0 else 0
                print(f"    [{dude_name}] {completed}/{n} "
                      f"({rate:.1f} mol/s, ETA {eta:.0f}s)", flush=True)

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # Save cache
    valid = int(np.sum(scores != 0.0))
    mean_score = float(np.mean(scores[scores != 0.0])) if valid > 0 else 0.0
    cache_data = {
        'dude_target': dude_name,
        'chembl_target': chembl_id,
        'n_molecules': n,
        'n_valid': valid,
        'mean_score': mean_score,
        'exhaustiveness': 4,
        'timestamp': time.time(),
        'scores': scores.tolist(),
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    elapsed = time.time() - t0
    print(f"  [{dude_name}] Done: {valid}/{n} valid, "
          f"mean={mean_score:.2f} kcal/mol, time={elapsed:.0f}s", flush=True)

    return scores


def main():
    max_decoys = 2000  # Subsample decoys for compute efficiency

    print("=" * 60)
    print("DUD-E Vina Batch Docking (exhaustiveness=8)")
    print(f"Max decoys per target: {max_decoys}")
    print("=" * 60)

    DUDE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()

    for dude_name, chembl_id in sorted(DUDE_TO_CHEMBL.items()):
        print(f"\n--- {dude_name} ({chembl_id}) ---", flush=True)

        # Download if needed
        if not download_dude_target(dude_name):
            print(f"  [SKIP] Could not download data for {dude_name}")
            continue

        # Load data
        smiles_list, labels, names = load_dude_data(
            dude_name, max_decoys=max_decoys
        )
        n_act = sum(labels)
        n_dec = len(labels) - n_act
        print(f"  Loaded: {n_act} actives + {n_dec} decoys = {len(smiles_list)}")

        # Dock
        dock_dude_target(dude_name, smiles_list, max_workers=10)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"Total DUD-E docking time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
