"""
Target Management Module for PureProtX

Defines benchmark targets (DUD-E, ChEMBL) and automates multi-target pipeline runs.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any


# DUD-E benchmark targets spanning 5 protein families
DUDE_TARGETS = {
    'AKT1':     {'family': 'Kinase',              'actives': 293,  'decoys': 16450,  'uniprot': 'P31749'},
    'EGFR':     {'family': 'Kinase',              'actives': 542,  'decoys': 35050,  'uniprot': 'P00533'},
    'ESR1_ago': {'family': 'Nuclear receptor',    'actives': 383,  'decoys': 20685,  'uniprot': 'P03372'},
    'PPARG':    {'family': 'Nuclear receptor',    'actives': 484,  'decoys': 25260,  'uniprot': 'P37231'},
    'DRD3':     {'family': 'GPCR',                'actives': 480,  'decoys': 34050,  'uniprot': 'P35462'},
    'ADRB2':    {'family': 'GPCR',                'actives': 231,  'decoys': 13550,  'uniprot': 'P07550'},
    'HDAC8':    {'family': 'Epigenetic',          'actives': 170,  'decoys': 10300,  'uniprot': 'Q9BY41'},
    'PDE5A':    {'family': 'Phosphodiesterase',   'actives': 398,  'decoys': 27600,  'uniprot': 'O76074'},
    'SRC':      {'family': 'Kinase',              'actives': 524,  'decoys': 34500,  'uniprot': 'P12931'},
    'VEGFR2':   {'family': 'Kinase',              'actives': 409,  'decoys': 24950,  'uniprot': 'P35968'},
}

# ChEMBL targets for benchmark study
CHEMBL_TARGETS = {
    'CHEMBL2487': {'name': 'Amyloid-beta A4 protein (APP)',  'family': 'Membrane protein'},
    'CHEMBL243':  {'name': 'HIV-1 Protease',                 'family': 'Protease'},
    'CHEMBL247':  {'name': 'HIV-1 Reverse Transcriptase',    'family': 'Reverse Transcriptase'},
    'CHEMBL279':  {'name': 'VEGFR2 (KDR)',                   'family': 'Kinase'},
    'CHEMBL3471': {'name': 'PI3K gamma',                     'family': 'Kinase'},
    'CHEMBL251':  {'name': 'Adenosine A2a receptor',         'family': 'GPCR'},
    'CHEMBL217':  {'name': 'Dopamine D2 receptor',           'family': 'GPCR'},
    'CHEMBL1862': {'name': 'Estrogen receptor alpha',        'family': 'Nuclear receptor'},
    'CHEMBL4005': {'name': 'PPARgamma',                      'family': 'Nuclear receptor'},
    'CHEMBL240':  {'name': 'hERG',                           'family': 'Ion channel'},
}

ALL_TARGETS = {**{k: {**v, 'source': 'dude'} for k, v in DUDE_TARGETS.items()},
               **{k: {**v, 'source': 'chembl'} for k, v in CHEMBL_TARGETS.items()}}


def get_target_info(target_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a target by ID."""
    return ALL_TARGETS.get(target_id)


def list_all_targets() -> pd.DataFrame:
    """List all available benchmark targets as a DataFrame."""
    rows = []
    for tid, info in ALL_TARGETS.items():
        row = {'target_id': tid, **info}
        rows.append(row)
    return pd.DataFrame(rows)


def get_dude_download_url(target_name: str) -> str:
    """Get DUD-E dataset download URL for a target."""
    return f"http://dude.docking.org/targets/{target_name.lower()}"


def prepare_dude_dataset(target_name: str, data_dir: str = 'data/dude') -> Dict[str, str]:
    """
    Prepare DUD-E dataset paths for a target.

    Expected DUD-E directory structure per target:
        data/dude/{target}/
            actives_final.sdf
            decoys_final.sdf
            receptor.pdb
            crystal_ligand.mol2

    Args:
        target_name: DUD-E target name (e.g., 'AKT1')
        data_dir: Base directory for DUD-E data

    Returns:
        Dict with file paths
    """
    target_dir = os.path.join(data_dir, target_name.lower())

    paths = {
        'actives': os.path.join(target_dir, 'actives_final.sdf'),
        'decoys': os.path.join(target_dir, 'decoys_final.sdf'),
        'receptor': os.path.join(target_dir, 'receptor.pdb'),
        'crystal_ligand': os.path.join(target_dir, 'crystal_ligand.mol2'),
    }

    # Check which files exist
    status = {}
    for key, path in paths.items():
        status[key] = os.path.exists(path)

    return {**paths, 'status': status, 'target': target_name, 'dir': target_dir}


def generate_pipeline_script(targets: List[str],
                              model_type: str = 'both',
                              output_path: str = 'run_pipeline.sh') -> str:
    """
    Generate a shell script to run the full pipeline across multiple targets.

    Args:
        targets: List of target IDs to process
        model_type: 'regression', 'classification', or 'both'
        output_path: Path to save the generated script

    Returns:
        Path to generated script
    """
    lines = [
        '#!/bin/bash',
        '# PureProtX Multi-Target Pipeline',
        f'# Generated for {len(targets)} targets',
        '',
        'set -e  # Exit on error',
        '',
        f'TARGETS=({" ".join(targets)})',
        '',
        'for target in "${TARGETS[@]}"; do',
        '    echo "============================================"',
        '    echo "Processing target: $target"',
        '    echo "============================================"',
        '',
        '    # Step 1: Fetch data',
        '    python PureProt.py fetch-data $target',
        '',
        f'    # Step 2: Train models ({model_type})',
        f'    python PureProt.py train-model ${{target}}_prepared_data.csv --model-type {model_type}',
        '',
        '    # Step 3: Screen (if receptor available)',
        '    if [ -f "data/dude/${target,,}/receptor.pdb" ]; then',
        '        python PureProt.py dock-batch ${target}_prepared_data.csv \\',
        '            --receptor data/dude/${target,,}/receptor.pdb',
        '    fi',
        '',
        '    echo "Completed: $target"',
        '    echo ""',
        'done',
        '',
        'echo "All targets processed successfully!"',
    ]

    script = '\n'.join(lines)
    with open(output_path, 'w', newline='\n') as f:
        f.write(script)

    return output_path
