"""
Blockchain Auditor Module for PureProtX

This module provides comprehensive blockchain auditing capabilities,
hashing models, proteins, parameters, and results for complete reproducibility.
"""

import json
import hashlib
import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain.purechain_connector import PurechainConnector


class BlockchainAuditor:
    """
    Comprehensive blockchain auditor for PureProtX screening results.
    Provides complete audit trail including models, proteins, parameters, and results.
    """
    
    def __init__(self, rpc_url: str = None, chain_id: int = None, network: str = 'testnet'):
        """
        Initialize the blockchain auditor.
        
        Args:
            rpc_url: Blockchain RPC URL
            chain_id: Blockchain chain ID
            network: Network type ('testnet', 'mainnet', 'local')
        """
        self.connector = PurechainConnector(
            rpc_url=rpc_url,
            chain_id=chain_id,
            network=network
        )
        
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 hash as hex string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def calculate_parameters_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Calculate hash of screening parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            SHA-256 hash of parameters
        """
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(sorted_params.encode()).hexdigest()
    
    def create_comprehensive_audit_record(self, 
                                        molecule_id: str,
                                        smiles: str,
                                        results: Dict[str, Any],
                                        model_path: Optional[str] = None,
                                        protein_path: Optional[str] = None,
                                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive audit record for blockchain storage.
        
        Args:
            molecule_id: Unique identifier for molecule
            smiles: SMILES string of molecule
            results: Screening results
            model_path: Path to AI model file
            protein_path: Path to protein structure file
            parameters: Screening parameters
            
        Returns:
            Complete audit record dictionary
        """
        audit_record = {
            'molecule_id': molecule_id,
            'smiles': smiles,
            'timestamp': int(time.time()),
            'results': results,
            'software_version': 'PureProtX-1.0.0',
            'hashes': {}
        }
        
        # Hash AI model file if provided
        if model_path and os.path.exists(model_path):
            audit_record['hashes']['ai_model'] = self.calculate_file_hash(model_path)
            audit_record['ai_model_path'] = os.path.basename(model_path)
        
        # Hash protein file if provided
        if protein_path and os.path.exists(protein_path):
            audit_record['hashes']['protein_structure'] = self.calculate_file_hash(protein_path)
            audit_record['protein_path'] = os.path.basename(protein_path)
        
        # Hash parameters if provided
        if parameters:
            audit_record['hashes']['parameters'] = self.calculate_parameters_hash(parameters)
            audit_record['parameters'] = parameters
        
        # Hash the complete results
        results_json = json.dumps(results, sort_keys=True, separators=(',', ':'))
        audit_record['hashes']['results'] = hashlib.sha256(results_json.encode()).hexdigest()
        
        # Calculate master hash of entire audit record
        audit_json = json.dumps(audit_record, sort_keys=True, separators=(',', ':'))
        audit_record['master_hash'] = hashlib.sha256(audit_json.encode()).hexdigest()
        
        return audit_record
    
    def record_screening_result(self, audit_record: Dict[str, Any]) -> Tuple[str, str]:
        """
        Record screening result on blockchain.
        
        Args:
            audit_record: Complete audit record
            
        Returns:
            Tuple of (transaction_hash, job_id)
        """
        # Generate unique job ID
        job_id = f"{audit_record['molecule_id']}_{audit_record['timestamp']}"
        
        # Record on blockchain
        tx_hash = self.connector.record_screening_result(
            job_id=job_id,
            molecule_id=audit_record['molecule_id'],
            smiles=audit_record['smiles'],
            result_hash=audit_record['master_hash'],
            additional_data=json.dumps(audit_record['hashes'])
        )
        
        print(f"✓ Screening result recorded on blockchain")
        print(f"  Job ID: {job_id}")
        print(f"  Transaction Hash: {tx_hash}")
        print(f"  Master Hash: {audit_record['master_hash']}")
        
        return tx_hash, job_id
    
    def verify_screening_result(self, job_id: str, local_audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify screening result against blockchain record.
        
        Args:
            job_id: Job ID to verify
            local_audit_record: Local audit record to compare
            
        Returns:
            Verification result dictionary
        """
        try:
            # Retrieve from blockchain
            blockchain_data = self.connector.get_screening_result(job_id)
            
            if not blockchain_data:
                return {
                    'verified': False,
                    'error': 'Job ID not found on blockchain'
                }
            
            # Compare hashes
            blockchain_hash = blockchain_data.get('result_hash')
            local_hash = local_audit_record['master_hash']
            
            verification_result = {
                'verified': blockchain_hash == local_hash,
                'job_id': job_id,
                'blockchain_hash': blockchain_hash,
                'local_hash': local_hash,
                'timestamp': blockchain_data.get('timestamp'),
                'transaction_hash': blockchain_data.get('transaction_hash')
            }
            
            if verification_result['verified']:
                print(f"✓ Verification successful for job {job_id}")
            else:
                print(f"✗ Verification failed for job {job_id}")
                print(f"  Blockchain hash: {blockchain_hash}")
                print(f"  Local hash: {local_hash}")
            
            return verification_result
            
        except Exception as e:
            return {
                'verified': False,
                'error': str(e),
                'job_id': job_id
            }
    
    def get_audit_summary(self, audit_record: Dict[str, Any]) -> str:
        """
        Generate human-readable audit summary.
        
        Args:
            audit_record: Audit record
            
        Returns:
            Formatted audit summary string
        """
        summary = f"""
=== PureProtX Audit Summary ===
Molecule ID: {audit_record['molecule_id']}
SMILES: {audit_record['smiles']}
Timestamp: {time.ctime(audit_record['timestamp'])}
Software: {audit_record['software_version']}

Hashes:
"""
        
        for hash_type, hash_value in audit_record['hashes'].items():
            summary += f"  {hash_type}: {hash_value[:16]}...\n"
        
        summary += f"Master Hash: {audit_record['master_hash']}\n"
        
        if 'results' in audit_record:
            summary += f"\nResults:\n"
            for key, value in audit_record['results'].items():
                if isinstance(value, float):
                    summary += f"  {key}: {value:.4f}\n"
                else:
                    summary += f"  {key}: {value}\n"
        
        return summary
    
    def test_connection(self) -> bool:
        """
        Test blockchain connection.
        
        Returns:
            True if connection successful
        """
        return self.connector.test_connection()
