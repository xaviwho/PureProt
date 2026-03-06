"""
AutoDock Vina Docking Module for PureProtX

Real structure-based docking using AutoDock Vina 1.2.7.
Replaces the ligand-based drug-likeness proxy with genuine
protein-structure-aware binding energy estimation.
"""

import os
import json
import time
import tempfile
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Lazy imports for multiprocessing compatibility
_RDKIT_LOADED = False
_MEEKO_LOADED = False


def _ensure_rdkit():
    global _RDKIT_LOADED
    if not _RDKIT_LOADED:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        _RDKIT_LOADED = True


def _ensure_meeko():
    global _MEEKO_LOADED
    if not _MEEKO_LOADED:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        _MEEKO_LOADED = True


# ---------------------------------------------------------------------------
# Target configuration: PDB files, binding site coordinates, box sizes
# ---------------------------------------------------------------------------
# Binding sites are defined from co-crystallized ligand centroids.
# Box sizes are 22 A per side (standard for small-molecule docking).

TARGET_DOCKING_CONFIG = {
    'CHEMBL243': {
        'pdb': 'CHEMBL243_1hxw.pdb',
        'pdb_id': '1HXW',
        'ligand_resname': 'RIT',
        'center': (10.8, 23.1, 3.9),
        'box_size': (22, 22, 22),
        'description': 'HIV-1 Protease (ritonavir site)',
    },
    'CHEMBL247': {
        'pdb': 'CHEMBL247_3qip.pdb',
        'pdb_id': '3QIP',
        'ligand_resname': 'NVP',
        'center': (10.9, 13.7, 17.3),
        'box_size': (22, 22, 22),
        'description': 'HIV-1 RT (NNRTI site, nevirapine)',
    },
    'CHEMBL279': {
        'pdb': 'CHEMBL279_3vhe.pdb',
        'pdb_id': '3VHE',
        'ligand_resname': '42Q',
        'center': (-25.0, -1.1, -10.5),
        'box_size': (24, 24, 24),
        'description': 'VEGFR2 (DFG-out site)',
    },
    'CHEMBL3471': {
        'pdb': 'CHEMBL3471_3apf.pdb',
        'pdb_id': '3APF',
        'ligand_resname': 'BMW',
        'center': (44.2, 14.9, 31.4),
        'box_size': (22, 22, 22),
        'description': 'PI3Kgamma (ATP site)',
    },
    'CHEMBL2487': {
        'pdb': 'CHEMBL2487_4ivt.pdb',
        'pdb_id': '4IVT',
        'ligand_resname': 'VTI',
        'center': (22.0, 23.9, 0.3),
        'box_size': (22, 22, 22),
        'description': 'BACE1 / APP (active site)',
    },
    'CHEMBL251': {
        'pdb': 'CHEMBL251_4eiy.pdb',
        'pdb_id': '4EIY',
        'ligand_resname': 'ZMA',
        'center': (-0.4, 8.5, 17.1),
        'box_size': (22, 22, 22),
        'description': 'Adenosine A2A (ZM241385 site)',
    },
    'CHEMBL217': {
        'pdb': 'CHEMBL217_6cm4.pdb',
        'pdb_id': '6CM4',
        'ligand_resname': '8NU',
        'center': (9.9, 5.8, -9.6),
        'box_size': (22, 22, 22),
        'description': 'Dopamine D2 (orthosteric site)',
    },
    'CHEMBL1862': {
        'pdb': 'CHEMBL1862_3ert.pdb',
        'pdb_id': '3ERT',
        'ligand_resname': 'OHT',
        'center': (31.6, -1.6, 25.6),
        'box_size': (22, 22, 22),
        'description': 'ER alpha (4-OHT site)',
    },
    'CHEMBL4005': {
        'pdb': 'CHEMBL4005_2prg.pdb',
        'pdb_id': '2PRG',
        'ligand_resname': 'BRL',
        'center': (54.7, -21.0, 30.5),
        'box_size': (22, 22, 22),
        'description': 'PPARgamma (rosiglitazone site)',
    },
    'CHEMBL240': {
        'pdb': 'CHEMBL240_5va1_assembly.pdb',
        'pdb_id': '5VA1',
        'ligand_resname': None,
        'center': (73.1, 73.2, 77.2),
        'box_size': (24, 24, 24),
        'description': 'hERG (inner cavity, Y652/F656)',
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURES_DIR = _PROJECT_ROOT / 'structures'
PREPARED_DIR = STRUCTURES_DIR / 'prepared'
CACHE_DIR = _PROJECT_ROOT / 'docking_cache'
VINA_EXE = _PROJECT_ROOT / 'tools' / 'vina.exe'
MEEKO_PREP = None  # Will be set dynamically


def _find_mk_prepare_receptor() -> Optional[str]:
    """Find meeko's mk_prepare_receptor executable."""
    import shutil
    path = shutil.which('mk_prepare_receptor')
    if path:
        return path
    # Check common locations
    for candidate in [
        Path(r'C:\Users\Xavie\AppData\Local\Python\pythoncore-3.12-64\Scripts\mk_prepare_receptor.exe'),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------

def prepare_receptor(target_id: str) -> str:
    """
    Prepare receptor PDBQT for a target using meeko.

    Args:
        target_id: ChEMBL target ID

    Returns:
        Path to prepared receptor PDBQT file
    """
    config = TARGET_DOCKING_CONFIG[target_id]
    pdb_path = STRUCTURES_DIR / config['pdb']
    output_pdbqt = PREPARED_DIR / f'{target_id}_receptor.pdbqt'

    if output_pdbqt.exists():
        return str(output_pdbqt)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    mk_prep = _find_mk_prepare_receptor()
    if mk_prep is None:
        raise RuntimeError("mk_prepare_receptor not found")

    cmd = [
        mk_prep,
        '--read_pdb', str(pdb_path),
        '-o', str(output_pdbqt).replace('.pdbqt', ''),
        '-p',
        '--charge_model', 'gasteiger',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not output_pdbqt.exists():
        raise RuntimeError(
            f"Receptor preparation failed for {target_id}: {result.stderr}"
        )

    return str(output_pdbqt)


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------

def prepare_ligand_pdbqt(smiles: str) -> Optional[str]:
    """
    Convert SMILES to PDBQT string using meeko.

    Args:
        smiles: SMILES string

    Returns:
        PDBQT string or None if preparation fails
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result < 0:
        # Retry with random coords
        result = AllChem.EmbedMolecule(
            mol, AllChem.ETKDGv3(), randomSeed=42
        )
        if result < 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass  # Use unoptimized coords

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


# ---------------------------------------------------------------------------
# Single-molecule docking (designed for use in worker processes)
# ---------------------------------------------------------------------------

def _dock_single_molecule(args: Tuple) -> Tuple[int, float]:
    """
    Dock a single molecule with Vina. Used by ProcessPoolExecutor.

    Args:
        args: (index, smiles, receptor_pdbqt, center, box_size, vina_exe_path)

    Returns:
        (index, score) where score is kcal/mol or 0.0 on failure
    """
    idx, smiles, receptor_pdbqt, center, box_size, vina_path = args

    # Prepare ligand PDBQT
    pdbqt_str = prepare_ligand_pdbqt(smiles)
    if pdbqt_str is None:
        return (idx, 0.0)

    # Write to temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdbqt', delete=False, dir=str(CACHE_DIR)
        ) as lig_f:
            lig_f.write(pdbqt_str)
            lig_path = lig_f.name

        out_path = lig_path.replace('.pdbqt', '_out.pdbqt')

        cmd = [
            str(vina_path),
            '--receptor', receptor_pdbqt,
            '--ligand', lig_path,
            '--center_x', f'{center[0]:.1f}',
            '--center_y', f'{center[1]:.1f}',
            '--center_z', f'{center[2]:.1f}',
            '--size_x', f'{box_size[0]}',
            '--size_y', f'{box_size[1]}',
            '--size_z', f'{box_size[2]}',
            '--exhaustiveness', '4',
            '--num_modes', '1',
            '--cpu', '1',
            '--out', out_path,
            '--verbosity', '0',
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

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
        # Cleanup temp files
        for p in [lig_path, out_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Batch docking
# ---------------------------------------------------------------------------

def dock_batch(
    target_id: str,
    smiles_list: List[str],
    max_workers: int = 6,
    max_molecules: int = 0,
    cache_file: Optional[str] = None,
) -> np.ndarray:
    """
    Dock a list of molecules against a target receptor using Vina.

    Args:
        target_id: ChEMBL target ID
        smiles_list: List of SMILES strings
        max_workers: Number of parallel Vina workers
        max_molecules: Max molecules to dock (0 = all)
        cache_file: Path to cache file for storing/loading results

    Returns:
        Array of Vina docking scores (kcal/mol, more negative = better)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        if len(cached['scores']) == len(smiles_list):
            print(f"    [CACHE] Loaded {len(cached['scores'])} Vina scores for {target_id}")
            return np.array(cached['scores'])

    config = TARGET_DOCKING_CONFIG[target_id]

    # Prepare receptor
    receptor_pdbqt = prepare_receptor(target_id)
    center = config['center']
    box_size = config['box_size']

    # Determine which molecules to dock
    n_total = len(smiles_list)
    if max_molecules > 0 and n_total > max_molecules:
        # Deterministic sampling
        rng = np.random.RandomState(42)
        dock_indices = sorted(rng.choice(n_total, max_molecules, replace=False))
    else:
        dock_indices = list(range(n_total))

    n_dock = len(dock_indices)
    print(f"    Docking {n_dock}/{n_total} molecules for {target_id} "
          f"(receptor: {config['pdb_id']}, site: {config['description']})")

    # Prepare work items
    vina_path = str(VINA_EXE)
    work_items = [
        (i, smiles_list[i], receptor_pdbqt, center, box_size, vina_path)
        for i in dock_indices
    ]

    # Run docking in parallel
    scores = np.zeros(n_total)
    completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_dock_single_molecule, item): item[0]
            for item in work_items
        }

        for future in as_completed(futures):
            idx, score = future.result()
            scores[idx] = score
            completed += 1

            if completed % 50 == 0 or completed == n_dock:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (n_dock - completed) / rate if rate > 0 else 0
                print(f"    [{target_id}] {completed}/{n_dock} docked "
                      f"({rate:.1f} mol/s, ETA {eta:.0f}s)")

    # For molecules not docked (if sampling), interpolate with median
    if max_molecules > 0 and n_total > max_molecules:
        docked_scores = scores[dock_indices]
        valid_scores = docked_scores[docked_scores != 0.0]
        if len(valid_scores) > 0:
            median_score = np.median(valid_scores)
            for i in range(n_total):
                if i not in dock_indices:
                    scores[i] = median_score

    # Save cache
    if cache_file:
        cache_data = {
            'target_id': target_id,
            'pdb_id': config['pdb_id'],
            'n_molecules': n_total,
            'n_docked': n_dock,
            'timestamp': time.time(),
            'scores': scores.tolist(),
        }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    elapsed = time.time() - t0
    valid = np.sum(scores != 0.0)
    print(f"    [{target_id}] Complete: {valid}/{n_dock} valid scores, "
          f"mean={np.mean(scores[scores!=0.0]):.2f} kcal/mol, "
          f"time={elapsed:.0f}s")

    return scores


def prepare_all_receptors():
    """Prepare PDBQT receptor files for all 10 targets."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    for target_id in TARGET_DOCKING_CONFIG:
        try:
            pdbqt_path = prepare_receptor(target_id)
            print(f"  [OK] {target_id}: {pdbqt_path}")
        except Exception as e:
            print(f"  [FAIL] {target_id}: {e}")


if __name__ == '__main__':
    print("Preparing all receptors...")
    prepare_all_receptors()
