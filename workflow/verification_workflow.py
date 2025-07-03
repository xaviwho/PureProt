"""Verification Workflow Module

This module ties together the blockchain verification and AI molecular modeling
components to create a verifiable drug screening system.
"""

import json
import hashlib
import time
import sys
import os
from typing import Dict, List, Any, Optional

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain.purechain_connector import PurechainConnector
from modeling.molecular_modeling import ScreeningPipeline, MoleculeRepresentation, BindingAffinityModel, ToxicityModel


class VerifiableDrugScreening:
    """Main class for verifiable drug screening workflow."""
    
    def __init__(self, rpc_url: str):
        """Initialize the verifiable drug screening system.
        
        Args:
            rpc_url: Purechain RPC endpoint URL
        """
        self.blockchain = PurechainConnector(rpc_url)
        self.screening_pipeline = ScreeningPipeline()
        self.job_history = {}
        # Load contract info if it exists
        contract_info_path = os.path.join(os.path.dirname(__file__), '..', 'blockchain', 'deployed_contract.json')
        if os.path.exists(contract_info_path):
            self.blockchain.load_contract_from_file(contract_info_path)
        else:
            print("Warning: Deployed contract information not found. Please run the deployment script.")
        
    def connect_wallet(self) -> bool:
        """Connect to the blockchain wallet.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        return self.blockchain.connect_wallet()
    
    def run_screening_job(self, molecule_id: str, smiles: str, target_id: str = "default") -> Dict[str, Any]:
        """Run a screening job, including blockchain submission."""
        # Run the screening pipeline
        result = self.run_screening(molecule_id, smiles, target_id)
        
        if "error" in result:
            return result

        # Generate a unique job ID
        job_id = f"{molecule_id}-{int(time.time())}"
        result['job_id'] = job_id

        # Submit the result hash to the blockchain
        print(f"\nSubmitting result for job {job_id} to the blockchain...")
        try:
            blockchain_receipt = self.blockchain.record_result(result)
            if blockchain_receipt and blockchain_receipt.get("tx_hash"):
                print(f"✓ Result recorded on blockchain with tx_hash: {blockchain_receipt['tx_hash']}")
                result['verification'] = {
                    "tx_hash": blockchain_receipt["tx_hash"],
                    "result_id": blockchain_receipt.get("result_id"),
                    "status": "recorded"
                }
            else:
                result['verification'] = {"status": "failed"}
        except Exception as e:
            print(f"✗ Blockchain submission failed: {e}")
            result['verification'] = {"status": "failed", "error": str(e)}

        # Store job in history
        self.job_history[job_id] = result
        return result

    def run_screening(self, molecule_id: str, smiles: str = None, target_id: str = "default") -> Dict[str, Any]:
        """Run a virtual drug screening for a single molecule.
        
        Args:
            molecule_id: Identifier for the molecule
            smiles: Optional SMILES string representing the molecule structure
            target_id: Identifier for the protein target
            
        Returns:
            Dict[str, Any]: Screening results
        """
        try:
            print(f"\n🧪 Running screening for molecule: {molecule_id} against target: {target_id}")
            
            # Use the screening pipeline to run the screening
            # This is the expected behavior in the tests
            result = self.screening_pipeline.screen_molecule(molecule_id, smiles, target_id)
            
            # Create molecule representation for fingerprint generation
            molecule = MoleculeRepresentation(molecule_id, smiles)
            
            # Print a nice formatted summary of the results
            print("\n📊 SCREENING RESULTS:")
            print(f"Molecule: {molecule_id}")
            print(f"Target: {target_id}")
            print(f"Binding affinity: {result['binding_affinity']} kcal/mol")
            print(f"Toxicity score: {result['toxicity_score']} (0-1 scale, higher = more toxic)")
            
            # Get features for Lipinski assessment
            features = molecule.extract_features()
            
            # Drug-likeness assessment based on Lipinski's Rule of Five
            lipinski_violations = 0
            if features.get("mol_weight", 0) > 500: lipinski_violations += 1
            if features.get("logp", 0) > 5: lipinski_violations += 1
            if features.get("h_donors", 0) > 5: lipinski_violations += 1
            if features.get("h_acceptors", 0) > 10: lipinski_violations += 1
            
            print(f"Lipinski violations: {lipinski_violations}/4 (0-1 is favorable)")
            
            # General assessment
            binding_affinity = result['binding_affinity']
            toxicity_score = result['toxicity_score']
            
            if binding_affinity < -8.0 and toxicity_score < 0.3 and lipinski_violations <= 1:
                assessment = "FAVORABLE - Strong binding, low toxicity, drug-like properties"
            elif binding_affinity < -7.0 and toxicity_score < 0.4 and lipinski_violations <= 1:
                assessment = "PROMISING - Good binding, acceptable toxicity profile"
            elif binding_affinity < -6.0 and toxicity_score < 0.5:
                assessment = "MODERATE - Acceptable binding, consider toxicity risks"
            else:
                assessment = "UNFAVORABLE - Weak binding and/or high toxicity concerns"
                
            print(f"Overall assessment: {assessment}")
            
            # Generate molecular fingerprint hash for verification
            mol_hash = molecule.get_molecular_hash()
            
            # Generate timestamp for this screening
            timestamp = int(time.time())
            
            # Compile results
            result = {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "target_id": target_id,
                "features": features,
                "binding_affinity": binding_affinity,
                "toxicity_score": toxicity_score,
                "lipinski_violations": lipinski_violations,
                "assessment": assessment,
                "timestamp": timestamp
            }
            
            return result
        except Exception as e:
            print(f"Error in screening: {e}")
            return {"error": str(e)}
    
    def batch_screen_molecules(self, molecules: Dict[str, str], target_id: str = "default") -> Dict[str, Dict[str, Any]]:
        """Run screening for multiple molecules.
        
        Args:
            molecules: Dictionary mapping molecule IDs to SMILES strings
            target_id: Identifier for the target protein
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of molecule IDs to results
        """
        results = {}
        for molecule_id, smiles in molecules.items():
            result = self.run_screening_job(molecule_id, smiles, target_id)
            results[molecule_id] = result
        return results
    
    def verify_screening(self, job_id: str) -> bool:
        """Verify a screening result using its on-chain result ID.

        Args:
            job_id: The job ID to verify.

        Returns:
            bool: True if verified, False otherwise.
        """
        try:
            # Check if job exists in history and has verification data
            job_data = self.job_history.get(job_id)
            if not job_data or "verification" not in job_data:
                print(f"No verification data found for job {job_id}")
                return False

            result_id = job_data["verification"].get("result_id")
            if not result_id:
                print(f"No on-chain Result ID found for job {job_id}")
                return False

            # Verify with blockchain using the result_id
            verified = self.blockchain.verify_result(result_id)

            # Print verification status
            molecule_id = job_data.get("molecule_id", job_id)
            tx_hash = job_data["verification"].get("tx_hash", "N/A")
            if verified:
                print(f"✓ Screening result for {molecule_id} (Job ID: {job_id}) successfully verified on blockchain")
                print(f"  - Transaction Hash: {tx_hash}")
            else:
                print(f"✗ Screening result for {molecule_id} (Job ID: {job_id}) verification failed")

            return verified
        except Exception as e:
            print(f"Error verifying result: {e}")
            return False
            
    
    def get_job_history(self) -> Dict[str, Dict[str, Any]]:
        """Get the full job history.
        
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of job IDs to job results
        """
        return self.job_history
    
    def save_results(self, file_path: str) -> bool:
        """Save all job results to a JSON file.
        
        Args:
            file_path: Path to save the results file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.job_history, f, indent=2)
            print(f"Results saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def load_results(self, file_path: str) -> bool:
        """Load job results from a JSON file.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, "r") as f:
                self.job_history = json.load(f)
            print(f"Results loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False


# Example usage
def main():
    """Main function to demonstrate the verifiable screening workflow."""
    print("Starting Verifiable Drug Screening Workflow...")
    
    # Initialize with Purechain RPC endpoint
    workflow = VerifiableDrugScreening("http://43.200.53.250:8548")
    
    # Connect to blockchain wallet by providing private key
    if not workflow.connect_wallet():
        print("\n✗ Wallet connection failed. Please check your private key and RPC endpoint.")
        return

    print("\n--- Running Single Screening Job ---")
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    job_result = workflow.run_screening_job("aspirin", aspirin_smiles)
    
    if job_result.get("verification", {}).get("status") == "recorded":
        print("\n--- Verifying Single Screening Job ---")
        job_id = job_result["job_id"]
        
        # Use the primary verification method, which now only needs the job_id
        workflow.verify_screening(job_id)
    else:
        print("\n✗ Skipping verification due to blockchain submission failure.")

    print("\n--- Running Batch Screening Job ---")
    batch_molecules = {
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "paracetamol": "CC(=O)NC1=CC=C(C=C1)O"
    }
    batch_results = workflow.batch_screen_molecules(batch_molecules)
    print(f"\nBatch screening completed for {len(batch_results)} molecules.")

    # Save all results to a file
    print("\n--- Saving All Results ---")
    workflow.save_results("screening_results.json")
    print("\nWorkflow finished.")


if __name__ == "__main__":
    main()
