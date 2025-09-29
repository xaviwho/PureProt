"""
Docking Engine Module for PureProtX

This module provides molecular docking capabilities with comprehensive
parameter tracking and result normalization for hybrid screening.
"""

import os
import tempfile
import subprocess
import platform
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.advanced_docking_engine import AdvancedDockingEngine, create_docking_engine


class DockingEngine:
    """
    Molecular docking engine with parameter tracking and result normalization.
    """
    
    def __init__(self, protein_path: Optional[str] = None):
        """
        Initialize the docking engine.
        
        Args:
            protein_path: Path to prepared protein structure file
        """
        self.protein_path = protein_path
        self.docking_engine = create_docking_engine()
        self.default_parameters = {
            'exhaustiveness': 8,
            'num_modes': 9,
            'energy_range': 3.0,
            'center': None,
            'size': [20, 20, 20]
        }
        
    def prepare_protein(self, pdb_path: str, output_path: str = None) -> str:
        """
        Prepare protein structure for docking.
        
        Args:
            pdb_path: Path to PDB file
            output_path: Output path for prepared protein
            
        Returns:
            Path to prepared protein file
        """
        if output_path is None:
            output_path = pdb_path.replace('.pdb', '_prepared.pdbqt')
        
        # Use the advanced docking engine for protein preparation
        prepared_path = self.docking_engine.prepare_protein(pdb_path, output_path)
        self.protein_path = prepared_path
        
        print(f"✓ Protein prepared: {prepared_path}")
        return prepared_path
    
    def find_binding_site(self, method: str = 'cavities') -> Dict[str, Any]:
        """
        Automatically detect binding site center coordinates.
        
        Args:
            method: Method for binding site detection
            
        Returns:
            Dictionary with binding site information
        """
        if not self.protein_path:
            raise ValueError("Protein path not set. Use prepare_protein() first.")
        
        binding_site = self.docking_engine.find_binding_site(self.protein_path, method)
        
        print(f"✓ Binding site detected: {binding_site}")
        return binding_site
    
    def dock_molecule(self, 
                     smiles: str, 
                     molecule_id: str,
                     center: List[float],
                     size: List[float] = None,
                     exhaustiveness: int = 8) -> Dict[str, Any]:
        """
        Dock a single molecule.
        
        Args:
            smiles: SMILES string of molecule
            molecule_id: Unique identifier for molecule
            center: Binding site center coordinates [x, y, z]
            size: Search box size [x, y, z]
            exhaustiveness: Docking exhaustiveness
            
        Returns:
            Docking results dictionary
        """
        if not self.protein_path:
            raise ValueError("Protein path not set. Use prepare_protein() first.")
        
        if size is None:
            size = self.default_parameters['size']
        
        # Prepare docking parameters
        docking_params = {
            'center': center,
            'size': size,
            'exhaustiveness': exhaustiveness,
            'num_modes': self.default_parameters['num_modes'],
            'energy_range': self.default_parameters['energy_range']
        }
        
        try:
            # Perform docking using advanced docking engine
            result = self.docking_engine.dock_molecule(
                smiles=smiles,
                protein_path=self.protein_path,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness
            )
            
            # Normalize and enhance results
            normalized_result = self._normalize_docking_result(result, molecule_id, docking_params)
            
            return normalized_result
            
        except Exception as e:
            return {
                'molecule_id': molecule_id,
                'smiles': smiles,
                'docking_score': None,
                'binding_affinity': None,
                'error': str(e),
                'success': False,
                'parameters': docking_params
            }
    
    def dock_batch(self, 
                   molecules_df: pd.DataFrame,
                   center: List[float],
                   size: List[float] = None,
                   exhaustiveness: int = 8) -> pd.DataFrame:
        """
        Dock a batch of molecules.
        
        Args:
            molecules_df: DataFrame with 'molecule_id' and 'smiles' columns
            center: Binding site center coordinates
            size: Search box size
            exhaustiveness: Docking exhaustiveness
            
        Returns:
            DataFrame with docking results
        """
        results = []
        
        for _, row in molecules_df.iterrows():
            result = self.dock_molecule(
                smiles=row['smiles'],
                molecule_id=row['molecule_id'],
                center=center,
                size=size,
                exhaustiveness=exhaustiveness
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _normalize_docking_result(self, 
                                 raw_result: Dict[str, Any], 
                                 molecule_id: str,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize docking results for consistency.
        
        Args:
            raw_result: Raw docking result
            molecule_id: Molecule identifier
            parameters: Docking parameters used
            
        Returns:
            Normalized result dictionary
        """
        normalized = {
            'molecule_id': molecule_id,
            'smiles': raw_result.get('smiles'),
            'docking_score': raw_result.get('docking_score'),
            'binding_affinity': raw_result.get('binding_affinity'),
            'success': raw_result.get('success', False),
            'parameters': parameters,
            'method': raw_result.get('method', 'unknown'),
            'poses': raw_result.get('poses', [])
        }
        
        # Convert binding affinity to normalized score (0-1 scale)
        if normalized['binding_affinity'] is not None:
            # Typical binding affinities range from 0 to -15 kcal/mol
            # Normalize to 0-1 scale where 1 is best (most negative)
            ba = float(normalized['binding_affinity'])
            normalized['normalized_score'] = max(0, min(1, (-ba) / 15.0))
        else:
            normalized['normalized_score'] = 0.0
        
        return normalized
    
    def combine_ai_docking_scores(self, 
                                 ai_score: float, 
                                 docking_score: float,
                                 ai_weight: float = 0.6,
                                 docking_weight: float = 0.4) -> Dict[str, float]:
        """
        Combine AI and docking scores using weighted consensus.
        
        Args:
            ai_score: AI prediction score (pIC50)
            docking_score: Normalized docking score (0-1)
            ai_weight: Weight for AI score
            docking_weight: Weight for docking score
            
        Returns:
            Dictionary with combined scores
        """
        # Normalize AI score to 0-1 scale (assuming pIC50 range 4-10)
        ai_normalized = max(0, min(1, (ai_score - 4.0) / 6.0))
        
        # Calculate weighted consensus
        consensus_score = (ai_weight * ai_normalized) + (docking_weight * docking_score)
        
        return {
            'ai_score_raw': ai_score,
            'ai_score_normalized': ai_normalized,
            'docking_score_normalized': docking_score,
            'consensus_score': consensus_score,
            'ai_weight': ai_weight,
            'docking_weight': docking_weight
        }
    
    def get_docking_parameters(self) -> Dict[str, Any]:
        """
        Get current docking parameters.
        
        Returns:
            Dictionary of docking parameters
        """
        return {
            'protein_path': self.protein_path,
            'default_parameters': self.default_parameters,
            'engine_info': self.docking_engine.get_engine_info() if hasattr(self.docking_engine, 'get_engine_info') else 'Advanced Docking Engine'
        }
    
    def validate_protein(self, protein_path: str) -> bool:
        """
        Validate protein structure file.
        
        Args:
            protein_path: Path to protein file
            
        Returns:
            True if valid
        """
        if not os.path.exists(protein_path):
            return False
        
        # Check file extension
        valid_extensions = ['.pdb', '.pdbqt']
        if not any(protein_path.endswith(ext) for ext in valid_extensions):
            return False
        
        # Basic file content validation
        try:
            with open(protein_path, 'r') as f:
                content = f.read(1000)  # Read first 1000 characters
                if protein_path.endswith('.pdb'):
                    return 'ATOM' in content or 'HETATM' in content
                elif protein_path.endswith('.pdbqt'):
                    return 'ATOM' in content and 'ROOT' in content
        except:
            return False
        
        return True
