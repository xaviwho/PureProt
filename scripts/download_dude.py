#!/usr/bin/env python3
"""
Download DUD-E benchmark data for PureProtX secondary evaluation.

Downloads actives and decoys SMILES for 5 DUD-E targets that correspond
to our ChEMBL benchmark targets. This enables a complementary evaluation
on scaffold-diverse data where docking should contribute more.

DUD-E mapping to ChEMBL targets:
  aa2ar  -> CHEMBL251  (Adenosine A2a, GPCR)
  esr1   -> CHEMBL1862 (ER alpha, Nuclear receptor)
  hivpr  -> CHEMBL243  (HIV-1 Protease, Protease)
  pparg  -> CHEMBL4005 (PPARgamma, Nuclear receptor)
  vgfr2  -> CHEMBL279  (VEGFR2, Kinase)
"""

import os
import sys
import csv
import gzip
import time
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUDE_DATA_DIR = PROJECT_ROOT / 'dude_data'

# DUD-E target name -> ChEMBL target ID mapping
DUDE_TO_CHEMBL = {
    'aa2ar': 'CHEMBL251',
    'esr1':  'CHEMBL1862',
    'hivpr': 'CHEMBL243',
    'pparg': 'CHEMBL4005',
    'vgfr2': 'CHEMBL279',
}

DUDE_BASE_URL = 'http://dude.docking.org/targets'


def download_file(url, dest_path, retries=3):
    """Download a file with retries."""
    for attempt in range(retries):
        try:
            print(f"    Downloading {url} ...", flush=True)
            urlretrieve(url, dest_path)
            return True
        except URLError as e:
            print(f"    Attempt {attempt+1} failed: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(5)
    return False


def parse_ism_file(filepath):
    """Parse a DUD-E .ism file (SMILES<tab>name format)."""
    smiles_list = []
    names = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                smiles_list.append(parts[0])
                names.append(parts[1])
            elif len(parts) == 1:
                smiles_list.append(parts[0])
                names.append(f'mol_{len(smiles_list)}')
    return smiles_list, names


def download_dude_target(dude_name):
    """Download actives and decoys for a single DUD-E target."""
    target_dir = DUDE_DATA_DIR / dude_name
    target_dir.mkdir(parents=True, exist_ok=True)

    actives_file = target_dir / 'actives_final.ism'
    decoys_file = target_dir / 'decoys_final.ism'

    # Check if already downloaded
    if actives_file.exists() and decoys_file.exists():
        act_smi, _ = parse_ism_file(actives_file)
        dec_smi, _ = parse_ism_file(decoys_file)
        print(f"  [CACHE] {dude_name}: {len(act_smi)} actives, "
              f"{len(dec_smi)} decoys", flush=True)
        return True

    # Download actives
    act_url = f'{DUDE_BASE_URL}/{dude_name}/actives_final.ism'
    if not download_file(act_url, str(actives_file)):
        print(f"  [FAIL] Could not download actives for {dude_name}")
        return False

    # Download decoys
    dec_url = f'{DUDE_BASE_URL}/{dude_name}/decoys_final.ism'
    if not download_file(dec_url, str(decoys_file)):
        print(f"  [FAIL] Could not download decoys for {dude_name}")
        return False

    # Verify
    act_smi, _ = parse_ism_file(actives_file)
    dec_smi, _ = parse_ism_file(decoys_file)
    print(f"  [OK] {dude_name}: {len(act_smi)} actives, "
          f"{len(dec_smi)} decoys", flush=True)
    return True


def load_dude_data(dude_name, max_decoys=0):
    """Load DUD-E SMILES and labels for a target.

    Args:
        dude_name: DUD-E target name (e.g. 'aa2ar')
        max_decoys: Max decoys to load (0 = all). If set, uses
                    deterministic random sampling.

    Returns:
        (smiles_list, labels, names) where labels are 1=active, 0=decoy
    """
    target_dir = DUDE_DATA_DIR / dude_name
    actives_file = target_dir / 'actives_final.ism'
    decoys_file = target_dir / 'decoys_final.ism'

    if not actives_file.exists() or not decoys_file.exists():
        raise FileNotFoundError(f"DUD-E data not found for {dude_name}. "
                                f"Run download_dude.py first.")

    act_smi, act_names = parse_ism_file(actives_file)
    dec_smi, dec_names = parse_ism_file(decoys_file)

    # Subsample decoys if requested
    if max_decoys > 0 and len(dec_smi) > max_decoys:
        import numpy as np
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(dec_smi), max_decoys, replace=False))
        dec_smi = [dec_smi[i] for i in idx]
        dec_names = [dec_names[i] for i in idx]

    all_smiles = act_smi + dec_smi
    all_names = act_names + dec_names
    labels = [1] * len(act_smi) + [0] * len(dec_smi)

    return all_smiles, labels, all_names


def main():
    print("=" * 60)
    print("DUD-E Data Download for PureProtX")
    print(f"Data directory: {DUDE_DATA_DIR}")
    print("=" * 60)

    DUDE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dude_name, chembl_id in DUDE_TO_CHEMBL.items():
        print(f"\n--- {dude_name} (ChEMBL: {chembl_id}) ---", flush=True)
        download_dude_target(dude_name)

    print("\nDone.")


if __name__ == '__main__':
    main()
