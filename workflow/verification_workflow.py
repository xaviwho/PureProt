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
    
    def __init__(self, rpc_url: str, chain_id: int, model_path: Optional[str] = None):
        """Initialize the verifiable drug screening system.

        Args:
            rpc_url: The RPC URL of the blockchain
            chain_id: The chain ID of the blockchain
            model_path: Optional path to a custom-trained AI model.
        """
        # Load environment variables from .env file
        load_dotenv()
        private_key = os.getenv("TEST_PRIVATE_KEY")
        if not private_key:
            raise ValueError("TEST_PRIVATE_KEY not found in .env file or environment variables.")

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

        self.blockchain_connector = PurechainConnector(
            rpc_url=rpc_url,
            private_key=private_key,
            chain_id=chain_id,
            contract_address=contract_address
        )

        self.model_path = model_path
        self._screening_pipeline = None
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

        # Create a clean copy of the result for hashing and storage.
        # This ensures that benchmark-related modifications do not affect verification.
        result_for_storage = result.copy()

        # Pre-calculate hashes from the clean data
        result_json = json.dumps(result_for_storage, sort_keys=True)
        result_hash_bytes = hashlib.sha256(result_json.encode()).digest()
        molecule_data_hash_bytes = hashlib.sha256(json.dumps(result_for_storage['molecule_data'], sort_keys=True).encode()).digest()

        # Submit the result hash to the blockchain
        print(f"\nSubmitting result for job {job_id} to the blockchain...")
        try:
            blockchain_response = self.blockchain_connector.record_and_verify_result(
                result_hash_bytes,
                molecule_data_hash_bytes,
                result_for_storage['molecule_id']
            )
            
            # Modify the live `result` object for CLI output, but not the stored one.
            result['ai_duration'] = result.pop('execution_time', 0)
            result['blockchain_duration'] = blockchain_response.get('duration', 0)

            if blockchain_response and blockchain_response.get("success"):
                tx_hash = blockchain_response['tx_hash']
                print(f"Success: Result recorded on blockchain with tx_hash: {tx_hash}")
                verification_data = {
                    "tx_hash": tx_hash,
                    "result_id": blockchain_response.get("numeric_id"),
                    "status": "recorded"
                }
            else:
                error_message = blockchain_response.get('error', 'Unknown error')
                print(f"Failed: Blockchain submission failed: {error_message}")
                verification_data = {"status": "failed", "error": error_message}
        except Exception as e:
            print(f"Failed: Blockchain submission failed: {e}")
            verification_data = {"status": "failed", "error": str(e)}

        # Add verification data to both the live object and the one for storage
        result['verification'] = verification_data
        result_for_storage['verification'] = verification_data

        # Store the clean, verifiable data in history
        self.job_history[job_id] = result_for_storage
        
        # Return the full result with benchmark data for immediate display
        return result

    def _get_screening_pipeline(self) -> ScreeningPipeline:
        """Lazily initialize and return the screening pipeline."""
        if self._screening_pipeline is None:
            print(f"Initializing screening pipeline with model: {self.model_path}")
            self._screening_pipeline = ScreeningPipeline(model_path=self.model_path)
        return self._screening_pipeline

    def run_screening(self, molecule_id: str, smiles: str = None, target_id: str = "default") -> Dict[str, Any]:
        """Run a virtual drug screening for a single molecule.

        Args:
            molecule_id: Identifier for the molecule
            smiles: Optional SMILES string representing the molecule structure
            target_id: Identifier for the protein target

        Returns:
            Dict[str, Any]: Screening results
        """
        start_time = time.time()
        try:
            print(f"\nRunning molecular modeling for molecule: {molecule_id} against target: {target_id}\n")

            # Lazily get the screening pipeline and run predictions
            pipeline = self._get_screening_pipeline()
            screening_result = pipeline.screen_molecule(molecule_id, smiles, target_id)

            if 'error' in screening_result:
                raise Exception(screening_result['error'])

            pIC50 = screening_result.get('predicted_pIC50', 0)
            lipinski_violations = screening_result.get('lipinski_violations', 5)

            # Print a nice formatted summary of the results
            print("\nSCREENING RESULTS:")
            print(f"Molecule: {molecule_id}")
            print(f"Target: {target_id}")
            print(f"Predicted pIC50: {pIC50:.4f}")
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
                "model_version": pipeline.binding_affinity_model.get_version(),
                "molecule_data": {
                    "smiles": smiles,
                    "lipinski_violations": lipinski_violations
                },
                "screening_data": {
                    "predicted_pIC50": pIC50,
                    "assessment": assessment
                }
            }

            end_time = time.time()
            execution_time = end_time - start_time
            final_result['execution_time'] = execution_time

            return final_result
        except Exception as e:
            print(f"Error in screening: {e}")
            return {"error": str(e), 'execution_time': time.time() - start_time}
    
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
        start_time = time.time()
        # 1. Load the local result from history
        local_result = self.job_history.get(job_id)
        if not local_result:
            return {"verified": False, "reason": f"Job ID {job_id} not found in local results.", 'execution_time': time.time() - start_time}

        # 2. Recalculate the hash from the local result data.
        # The dictionary that was originally hashed is the entire result object,
        # minus the 'verification' key that is added after hashing.
        result_to_hash = {k: v for k, v in local_result.items() if k != 'verification'}
        result_json = json.dumps(result_to_hash, sort_keys=True)
        local_hash_bytes = hashlib.sha256(result_json.encode()).digest()

        # 3. Call the blockchain connector to verify the hash
        verification_result = self.blockchain_connector.verify_result_client_side(tx_hash, local_hash_bytes)
        end_time = time.time()
        execution_time = end_time - start_time
        verification_result['execution_time'] = execution_time
        return verification_result

    def verify_result_from_history(self, job_id: str) -> Dict[str, Any]:
        """Verify a result by loading it from history and using its tx_hash."""
        local_result = self.job_history.get(job_id)
        if not local_result:
            return {"verified": False, "reason": f"Job ID {job_id} not found in local history."}

        verification_data = local_result.get("verification")
        if not verification_data or "tx_hash" not in verification_data or not verification_data["tx_hash"]:
            return {"verified": False, "reason": f"Transaction hash not found for Job ID {job_id}."}

        tx_hash = verification_data["tx_hash"]
        print(f"Verifying Job ID: {job_id} with TX Hash: {tx_hash}")
        return self.verify_result(job_id, tx_hash)
            
    
    def show_history(self):
        """Prints a summary of the job history."""
        print("\n--- Screening Job History ---")
        if not self.job_history:
            print("No jobs found in history.")
            return

        for job_id, result in self.job_history.items():
            status = result.get('verification', {}).get('status', 'unknown')
            pIC50_data = result.get('screening_data', {}).get('predicted_pIC50')
            molecule_id = result.get('molecule_id', 'N/A')
            
            print(f"  Job ID: {job_id}")
            print(f"    Molecule: {molecule_id}")
            if pIC50_data is not None:
                print(f"    pIC50: {pIC50_data:.4f}")
            else:
                print(f"    pIC50: N/A")
            print(f"    Status: {status}")
            print("-" * 20)

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
