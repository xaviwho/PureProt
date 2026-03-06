#!/usr/bin/env python3
"""Re-dock CHEMBL1862 val set — previous run had only 17/1031 valid scores.

Run this AFTER the main ChEMBL batch docking finishes to avoid CPU contention.
Uses parallel workers with proper Windows multiprocessing guard.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from scripts.run_vina_docking import (
        load_and_split_data, dock_single, CACHE_DIR, TARGET_CONFIG, VINA_EXE, PREPARED_DIR
    )

    target_id = 'CHEMBL1862'
    set_name = 'val'

    # Delete the broken cache
    cache_file = CACHE_DIR / f'{target_id}_{set_name}_vina_e4.json'
    if cache_file.exists():
        cache_file.unlink()
        print(f"Deleted broken cache: {cache_file}")
    else:
        print(f"No existing cache to delete (already removed)")

    # Load and split to get val SMILES
    train_smi, val_smi, test_smi = load_and_split_data(target_id)
    print(f"{target_id}: Train={len(train_smi)}, Val={len(val_smi)}, Test={len(test_smi)}")

    # Set up docking params
    cfg = TARGET_CONFIG[target_id]
    receptor = os.path.join(PREPARED_DIR, f'{target_id}_receptor.pdbqt')
    center = cfg['center']
    box = cfg['box']

    # Create temp directory
    tmp_dir = str(CACHE_DIR / f'tmp_{target_id}_{set_name}_redock')
    os.makedirs(tmp_dir, exist_ok=True)

    n = len(val_smi)
    scores = np.zeros(n)
    t0 = time.time()
    max_workers = 8

    print(f"Docking {n} molecules with {max_workers} workers...", flush=True)

    # Prepare work items
    work_items = [
        (i, smi, receptor, center, box, VINA_EXE, tmp_dir)
        for i, smi in enumerate(val_smi)
    ]

    completed = 0
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
                n_valid = int(np.sum(scores != 0.0))
                print(f"  [{completed}/{n}] {n_valid} valid, "
                      f"{rate:.2f} mol/s, ETA {eta:.0f}s", flush=True)

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

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"\nDone: {valid}/{n} valid, mean={mean_score:.2f} kcal/mol, time={elapsed:.0f}s")


if __name__ == '__main__':
    main()
