"""Verification Workflow Module

This module ties together the blockchain verification and AI molecular modeling
components to create a verifiable drug screening system.
"""

import json
import hashlib
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blockchain.purechain_connector import PurechainConnector
from modeling.molecular_modeling import ScreeningPipeline


from dotenv import load_dotenv

class VerifiableDrugScreening:
    """Main class for verifiable drug screening workflow."""
    
    def __init__(self, rpc_url: str, chain_id: int):
        """Initialize the verifiable drug screening system.
        
        Args:
            rpc_url: Purechain RPC endpoint URL
            chain_id: The chain ID of the blockchain
        """
        # Load environment variables from .env file
        load_dotenv()
        private_key = os.getenv("TEST_PRIVATE_KEY")

        # Load contract address from deployment info file
        deployment_info_path = Path(__file__).parent.parent / "local_deployment_info.json"
        try:
            with open(deployment_info_path, "r") as f:
                deployment_info = json.load(f)
                contract_address = deployment_info.get("address")
                if not contract_address:
                    raise ValueError(f"Contract address not found in {deployment_info_path}")
        except Exception as e:
            print(f"Error loading contract address: {e}")
            raise

        self.blockchain = PurechainConnector(
            rpc_url=rpc_url,
            contract_address=contract_address,
            private_key=private_key,
            chain_id=chain_id
        )
        self.screening_pipeline = ScreeningPipeline()
        self.job_history = {}
        

    
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
            blockchain_response = self.blockchain.record_and_verify_result(result)
            if blockchain_response and blockchain_response.get("success"):
                tx_hash = blockchain_response['tx_hash']
                print(f"Success: Result recorded on blockchain with tx_hash: {tx_hash}")
                result['verification'] = {
                    "tx_hash": tx_hash,
                    "result_id": blockchain_response.get("numeric_id"),
                    "status": "recorded"
                }
            else:
                error_message = blockchain_response.get('error', 'Unknown error')
                print(f"Failed: Blockchain submission failed: {error_message}")
                result['verification'] = {"status": "failed"}
        except Exception as e:
            print(f"Failed: Blockchain submission failed: {e}")
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
            print(f"\nRunning molecular modeling for molecule: {molecule_id} against target: {target_id}")
            
            # Use the screening pipeline to run the screening
            result = self.screening_pipeline.screen_molecule(molecule_id, smiles, target_id)

            if 'error' in result:
                print(f"Screening failed: {result['error']}")
                return result

            # Extract results for assessment
            pIC50 = result.get('predicted_pIC50', 0)
            lipinski_violations = result.get('lipinski_violations', 5) # Default to high violations

            # Print a nice formatted summary of the results
            print("\nSCREENING RESULTS:")
            print(f"Molecule: {molecule_id}")
            print(f"Target: {target_id}")
            print(f"Predicted pIC50: {pIC50}")
            print(f"Lipinski Violations: {lipinski_violations}/4 (0-1 is favorable)")

            # General assessment based on available data
            if pIC50 > 7.0 and lipinski_violations <= 1:
                assessment = "PROMISING - Good binding affinity and drug-like properties"
            elif pIC50 > 6.0 and lipinski_violations <= 2:
                assessment = "MODERATE - Acceptable binding, some drug-likeness issues"
            else:
                assessment = "UNFAVORABLE - Weak binding and/or poor drug-like properties"

            print(f"Overall assessment: {assessment}")

            # Generate timestamp for this screening
            timestamp = int(time.time())

            # Compile final result object for blockchain record
            final_result = {
                "molecule_id": molecule_id,
                "target_id": target_id,
                "timestamp": timestamp,
                "model_version": self.screening_pipeline.binding_affinity_model.get_version(),
                "molecule_data": {
                    "smiles": smiles,
                    "lipinski_violations": lipinski_violations
                },
                "screening_data": {
                    "predicted_pIC50": pIC50,
                    "assessment": assessment
                }
            }

            return final_result
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

    def verify_result(self, job_id: str, tx_hash: str) -> Dict[str, Any]:
        """Verifies a screening result against the blockchain record."""
        # 1. Load the local result from history
        local_result = self.job_history.get(job_id)
        if not local_result:
            return {"verified": False, "reason": f"Job ID {job_id} not found in local results."}

        # 2. Recalculate the hash from the local result data.
        # The dictionary that was originally hashed is the entire result object,
        # minus the 'verification' key that is added after hashing.
        result_to_hash = {k: v for k, v in local_result.items() if k != 'verification'}
        result_json = json.dumps(result_to_hash, sort_keys=True)
        local_hash_bytes = hashlib.sha256(result_json.encode()).digest()

        # 3. Call the blockchain connector to verify the hash
        return self.blockchain.verify_result_client_side(tx_hash, local_hash_bytes)
            
    
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
