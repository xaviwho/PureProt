"""
Data Manager Module for PureProtX

This module handles data fetching, preparation, and management
for molecular datasets from ChEMBL and other sources.
"""

import os
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.data_loader import fetch_and_prepare_data


class DataManager:
    """
    Data manager for fetching and preparing molecular datasets.
    """
    
    def __init__(self):
        """Initialize the data manager."""
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.supported_targets = {
            'CHEMBL243': 'HIV-1 Protease',
            'CHEMBL4822': 'BRAF',
            'CHEMBL203': 'EGFR',
            'CHEMBL279': 'Cathepsin S',
            'CHEMBL5145': 'BRAF (duplicate)',
            'CHEMBL2111353': 'Unknown Target',
            'CHEMBL3638323': 'Unknown Target',
            'CHEMBL3713916': 'Unknown Target',
            'CHEMBL3736': 'Unknown Target'
        }
    
    def fetch_chembl_data(self, target_id: str, output_path: str = None) -> str:
        """
        Fetch and prepare data from ChEMBL for a specific target.
        
        Args:
            target_id: ChEMBL target identifier
            output_path: Output path for prepared data
            
        Returns:
            Path to prepared dataset file
        """
        if output_path is None:
            output_path = f"{target_id.lower()}_prepared_data.csv"
        
        print(f"Fetching data for target: {target_id}")
        
        if target_id in self.supported_targets:
            print(f"Target: {self.supported_targets[target_id]}")
        
        try:
            # Use existing data loader functionality
            dataset_path = fetch_and_prepare_data(target_id, output_path)
            
            # Validate the prepared dataset
            self._validate_dataset(dataset_path)
            
            print(f"✓ Data prepared successfully: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            print(f"✗ Error fetching data for {target_id}: {e}")
            raise
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load a prepared dataset from file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        self._validate_dataset_format(df)
        
        return df
    
    def prepare_batch_molecules(self, 
                              molecules: List[Dict[str, str]], 
                              output_path: str = "batch_molecules.csv") -> str:
        """
        Prepare a batch of molecules for screening.
        
        Args:
            molecules: List of dictionaries with 'molecule_id' and 'smiles'
            output_path: Output path for batch file
            
        Returns:
            Path to prepared batch file
        """
        df = pd.DataFrame(molecules)
        
        # Validate required columns
        if 'molecule_id' not in df.columns or 'smiles' not in df.columns:
            raise ValueError("Molecules must have 'molecule_id' and 'smiles' columns")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['smiles'])
        
        # Validate SMILES
        valid_molecules = []
        for _, row in df.iterrows():
            if self._validate_smiles(row['smiles']):
                valid_molecules.append(row.to_dict())
            else:
                print(f"Warning: Invalid SMILES for {row['molecule_id']}: {row['smiles']}")
        
        if not valid_molecules:
            raise ValueError("No valid molecules found in batch")
        
        # Save prepared batch
        valid_df = pd.DataFrame(valid_molecules)
        valid_df.to_csv(output_path, index=False)
        
        print(f"✓ Batch prepared: {len(valid_molecules)} valid molecules saved to {output_path}")
        return output_path
    
    def convert_smi_to_csv(self, smi_path: str, output_path: str = None) -> str:
        """
        Convert SMILES file to PureProtX-compatible CSV format.
        
        Args:
            smi_path: Path to .smi file
            output_path: Output path for CSV file
            
        Returns:
            Path to converted CSV file
        """
        if output_path is None:
            output_path = smi_path.replace('.smi', '.csv')
        
        molecules = []
        
        with open(smi_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 1:
                        smiles = parts[0]
                        mol_id = parts[1] if len(parts) > 1 else f"mol_{i+1}"
                        
                        if self._validate_smiles(smiles):
                            molecules.append({
                                'molecule_id': mol_id,
                                'smiles': smiles
                            })
        
        if not molecules:
            raise ValueError("No valid molecules found in SMILES file")
        
        df = pd.DataFrame(molecules)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Converted {len(molecules)} molecules from {smi_path} to {output_path}")
        return output_path
    
    def get_dataset_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Dictionary with dataset information
        """
        df = self.load_dataset(file_path)
        
        info = {
            'file_path': file_path,
            'num_molecules': len(df),
            'columns': list(df.columns),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }
        
        # Add statistics for numeric columns
        if 'pic50' in df.columns:
            info['pic50_stats'] = {
                'mean': df['pic50'].mean(),
                'std': df['pic50'].std(),
                'min': df['pic50'].min(),
                'max': df['pic50'].max()
            }
        
        return info
    
    def list_available_datasets(self, data_dir: str = ".") -> List[Dict[str, Any]]:
        """
        List available prepared datasets.
        
        Args:
            data_dir: Directory to search for datasets
            
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        for file in os.listdir(data_dir):
            if file.endswith('_prepared_data.csv') or file.endswith('_data.csv'):
                file_path = os.path.join(data_dir, file)
                try:
                    info = self.get_dataset_info(file_path)
                    datasets.append(info)
                except Exception as e:
                    print(f"Warning: Could not read dataset {file}: {e}")
        
        return datasets
    
    def _validate_dataset(self, file_path: str) -> bool:
        """
        Validate a prepared dataset file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            True if valid
        """
        try:
            df = pd.read_csv(file_path)
            self._validate_dataset_format(df)
            return True
        except Exception as e:
            raise ValueError(f"Invalid dataset format: {e}")
    
    def _validate_dataset_format(self, df: pd.DataFrame) -> None:
        """
        Validate dataset format requirements.
        
        Args:
            df: DataFrame to validate
        """
        required_columns = ['smiles']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        # Validate SMILES column
        invalid_smiles = 0
        for smiles in df['smiles'].head(10):  # Check first 10
            if not self._validate_smiles(smiles):
                invalid_smiles += 1
        
        if invalid_smiles > 5:  # More than 50% invalid in sample
            raise ValueError("Dataset contains too many invalid SMILES")
    
    def _validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_supported_targets(self) -> Dict[str, str]:
        """
        Get list of supported ChEMBL targets.
        
        Returns:
            Dictionary of target IDs and names
        """
        return self.supported_targets.copy()
