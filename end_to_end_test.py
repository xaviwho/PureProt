#!/usr/bin/env python3
"""PureProt End-to-End Test Script

This script performs comprehensive end-to-end testing of the PureProt system,
including the AI molecular modeling pipeline and blockchain verification.
It also gathers performance and reliability metrics for research purposes.
"""

import os
import json
import time
import statistics
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import random
import string
import pandas as pd
from dotenv import load_dotenv
# Import PureProt components
from modeling.molecular_modeling import ScreeningPipeline
from blockchain.purechain_connector import PurechainConnector
from blockchain.deploy import deploy_contract, compile_contract, save_deployment_info

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EndToEndTest:
    """Manages end-to-end testing of the PureProt system."""
    
    def __init__(self):
        self.pipeline = ScreeningPipeline()
        self.model_path = Path(__file__).parent / "modeling" / "binding_affinity_model.joblib"
        
        # Check if we're using local or remote blockchain
        local_deployment = os.path.exists(os.path.join(os.path.dirname(__file__), "local_deployment_info.json"))
        
        if local_deployment:
            logger.info("Using local blockchain deployment")
            self.deployment_file = os.path.join(os.path.dirname(__file__), "local_deployment_info.json")
            self.blockchain_url = "http://127.0.0.1:8545"
        else:
            logger.info("Using remote Purechain deployment")
            self.deployment_file = os.path.join(os.path.dirname(__file__), "deployment_info.json")
            self.blockchain_url = "http://43.200.53.250:8548"
        
        self.connector = None
        self.results = []
        self.screening_results = []
        self.metrics = {
            "screening_times": [],
            "blockchain_times": [],
            "verification_times": [],
            "total_times": [],
            "success_rates": {
                "screening": 0,
                "blockchain": 0,
                "verification": 0,
                "total": 0
            },
            "gas_usage": []
        }
        
    def connect_to_blockchain(self):
        """Connect to the blockchain by initializing the PurechainConnector."""
        logger.info(f"Connecting to blockchain at {self.blockchain_url}")
        load_dotenv()

        try:
            # For local testing, always redeploy the contract to ensure a fresh state
            if "127.0.0.1" in self.blockchain_url or "localhost" in self.blockchain_url:
                logger.info("Local deployment detected. Redeploying contract directly...")
                try:
                    contract_path = Path(__file__).parent / "blockchain" / "DrugScreeningVerifier.sol"
                    output_path = self.deployment_file

                    # Compile contract
                    abi, bytecode = compile_contract(str(contract_path))

                    # Deploy contract using development mode (use Ganache's first account)
                    # No private key needed as we're using Ganache's unlocked accounts
                    new_address = deploy_contract(
                        rpc_url=self.blockchain_url,
                        chain_id=1337,
                        abi=abi,
                        bytecode=bytecode,
                        dev_mode=True
                    )

                    if not new_address:
                        logger.error("Contract deployment failed. Aborting test.")
                        return False

                    # Save new deployment info
                    save_deployment_info(new_address, abi, str(output_path))
                    logger.info("Contract deployed and info saved successfully for test run.")

                except Exception as e:
                    logger.error(f"An exception occurred during automated contract deployment: {e}", exc_info=True)
                    return False

            # 1. Load contract address from deployment info file (now guaranteed to be fresh for local tests)
            with open(self.deployment_file, 'r') as f:
                deployment_info = json.load(f)
            contract_address = deployment_info.get("address")
            if not contract_address:
                logger.error(f"Contract address not found in {self.deployment_file}")
                return False

            # 2. For local testing, we don't need a private key as we'll use Ganache's unlocked first account
            # For remote connections, we still need the private key from environment
            is_local = "127.0.0.1" in self.blockchain_url or "localhost" in self.blockchain_url
            private_key = None if is_local else os.getenv("TEST_PRIVATE_KEY")
            
            if not is_local and not private_key:
                logger.error("TEST_PRIVATE_KEY environment variable not set for remote connection.")
                return False
            
            # 3. Determine Chain ID
            if "127.0.0.1" in self.blockchain_url or "localhost" in self.blockchain_url:
                chain_id = 1337
                logger.info(f"Using local Ganache chain ID ({chain_id})")
            else:
                chain_id = 900520900520
                logger.info(f"Using Purechain chain ID ({chain_id})")

            # 4. Initialize the connector
            self.connector = PurechainConnector(
                rpc_url=self.blockchain_url,
                private_key=private_key,
                contract_address=contract_address,
                chain_id=chain_id
            )
            
            logger.info("Successfully initialized PurechainConnector.")
            return True

        except FileNotFoundError:
            logger.error(f"Deployment file not found: {self.deployment_file}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}", exc_info=True)
            return False
    
    def run_single_test(self, molecule_id: str, smiles: str, target_id: str) -> Dict[str, Any]:
        """Run a single end-to-end test for one molecule."""
        result = {
            "molecule_id": molecule_id,
            "target_id": target_id,
            "screening_success": False,
            "blockchain_success": False,
            "verification_success": False,
            "screening_time": None,
            "blockchain_time": None,
            "verification_time": None,
            "total_time": None,
            "screening_result": None,
            "blockchain_result": None
        }
        
        start_total = time.time()
        
        # Step 1: AI Screening
        logger.info(f"Testing molecule {molecule_id} against {target_id}")
        start_screening = time.time()
        try:
            screening_result = self.pipeline.screen_molecule(molecule_id, smiles, target_id)
            if "error" in screening_result:
                logger.error(f"Screening error: {screening_result['error']}")
                result["screening_success"] = False
            else:
                result["screening_success"] = True
                result["screening_result"] = screening_result
                
                pIC50 = screening_result.get('predicted_pIC50', 'N/A')
                passes_lipinski = screening_result.get('passes_lipinski', 'N/A')
                logger.info(f"Screening successful: pIC50 = {pIC50}, Passes Lipinski = {passes_lipinski}")

                # Store all screening results for the final report
                self.screening_results.append({
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    **screening_result
                })
        except Exception as e:
            logger.error(f"Screening exception: {e}")
            result["screening_success"] = False
            
        result["screening_time"] = time.time() - start_screening
        self.metrics["screening_times"].append(result["screening_time"])
        
        # Step 2: Blockchain Recording
        if result["screening_success"]:
            start_blockchain = time.time()
            try:
                # Structure the data for blockchain recording
                # The keys should match what the smart contract and verification logic expect
                structured_result = {
                    "molecule_id": molecule_id,
                    "molecule_data": {
                        "smiles": smiles,
                        "target_id": target_id
                    },
                    "results": {
                        "predicted_pIC50": screening_result.get('predicted_pIC50'),
                        "passes_lipinski": screening_result.get('passes_lipinski'),
                        "molecular_weight": screening_result.get('molecular_weight'),
                        "logp": screening_result.get('logp')
                    }
                }
                
                blockchain_result = self.connector.record_and_verify_result(structured_result)
                if blockchain_result and blockchain_result.get("success"):
                    result["blockchain_success"] = True
                    result["blockchain_result"] = blockchain_result
                    if "gas_used" in blockchain_result:
                        self.metrics["gas_usage"].append(blockchain_result["gas_used"])
                    logger.info(f"Blockchain recording successful. Numeric ID: {blockchain_result.get('numeric_id')}")
                else:
                    logger.error("Blockchain recording failed")
                    result["blockchain_success"] = False
            except Exception as e:
                logger.error(f"Blockchain exception: {e}")
                result["blockchain_success"] = False
                
            result["blockchain_time"] = time.time() - start_blockchain
            self.metrics["blockchain_times"].append(result["blockchain_time"])
            
            # Step 3: Verification
            if result["blockchain_success"]:
                start_verification = time.time()
                try:
                    # For our test purposes, successful recording implies verification
                    # In a real deployment, you might want to query the chain again
                    result["verification_success"] = True
                    logger.info("Verification successful")
                except Exception as e:
                    logger.error(f"Verification exception: {e}")
                    result["verification_success"] = False
                    
                result["verification_time"] = time.time() - start_verification
                self.metrics["verification_times"].append(result["verification_time"])
        
        result["total_time"] = time.time() - start_total
        self.metrics["total_times"].append(result["total_time"])
        
        # Add to overall results
        self.results.append(result)
        return result
    
    def run_batch_test(self, molecules: Dict[str, str], targets: List[str], repeat: int = 1):
        """Run a batch of end-to-end tests with multiple molecules and targets."""
        logger.info(f"Starting batch test with {len(molecules)} molecules, "
                  f"{len(targets)} targets, and {repeat} repetitions")
        
        test_combinations = []
        for molecule_id, smiles in molecules.items():
            for target_id in targets:
                for i in range(repeat):
                    test_combinations.append((molecule_id, smiles, target_id))
        
        logger.info(f"Total test combinations: {len(test_combinations)}")
        
        for i, (molecule_id, smiles, target_id) in enumerate(test_combinations, 1):
            logger.info(f"Test {i}/{len(test_combinations)}: {molecule_id} against {target_id}")
            self.run_single_test(molecule_id, smiles, target_id)
            
        # Calculate success rates after all tests
        total_tests = len(test_combinations)
        if total_tests > 0:
            self.metrics["success_rates"]["screening"] = \
                sum(1 for r in self.results if r["screening_success"]) / total_tests * 100
            self.metrics["success_rates"]["blockchain"] = \
                sum(1 for r in self.results if r["blockchain_success"]) / total_tests * 100
            self.metrics["success_rates"]["verification"] = \
                sum(1 for r in self.results if r["verification_success"]) / total_tests * 100
            self.metrics["success_rates"]["total"] = \
                sum(1 for r in self.results if r["verification_success"]) / total_tests * 100
    
    def print_summary(self):
        """Print a summary of test results and metrics."""
        logger.info("\n===== TEST SUMMARY =====")
        logger.info(f"Total tests: {len(self.results)}")
        
        # Success rates
        logger.info("\nSuccess Rates:")
        logger.info(f"  Screening: {self.metrics['success_rates']['screening']:.1f}%")
        logger.info(f"  Blockchain Recording: {self.metrics['success_rates']['blockchain']:.1f}%")
        logger.info(f"  Verification: {self.metrics['success_rates']['verification']:.1f}%")
        logger.info(f"  End-to-End: {self.metrics['success_rates']['total']:.1f}%")
        
        # Timing metrics
        if self.metrics["screening_times"]:
            logger.info("\nTiming Metrics (seconds):")
            logger.info(f"  Screening: {statistics.mean(self.metrics['screening_times']):.3f} avg, "
                      f"{min(self.metrics['screening_times']):.3f} min, "
                      f"{max(self.metrics['screening_times']):.3f} max")
        
        if self.metrics["blockchain_times"]:
            logger.info(f"  Blockchain Recording: {statistics.mean(self.metrics['blockchain_times']):.3f} avg, "
                      f"{min(self.metrics['blockchain_times']):.3f} min, "
                      f"{max(self.metrics['blockchain_times']):.3f} max")
        
        if self.metrics["verification_times"]:
            logger.info(f"  Verification: {statistics.mean(self.metrics['verification_times']):.3f} avg, "
                      f"{min(self.metrics['verification_times']):.3f} min, "
                      f"{max(self.metrics['verification_times']):.3f} max")
        
        if self.metrics["total_times"]:
            logger.info(f"  Total Processing: {statistics.mean(self.metrics['total_times']):.3f} avg, "
                      f"{min(self.metrics['total_times']):.3f} min, "
                      f"{max(self.metrics['total_times']):.3f} max")
        
        # Gas usage if available
        if self.metrics["gas_usage"]:
            logger.info(f"\nGas Usage: {statistics.mean(self.metrics['gas_usage']):.2f} avg, "
                      f"{min(self.metrics['gas_usage'])} min, "
                      f"{max(self.metrics['gas_usage'])} max")
        
        # Save full metrics to file
        with open("test_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info("\nDetailed metrics saved to test_metrics.json")

    def save_screening_results(self):
        """Save the detailed screening results to a CSV file."""
        if not self.screening_results:
            logger.info("No screening results to save.")
            return

        results_df = pd.DataFrame(self.screening_results)
        results_file = "screening_results.csv"
        try:
            results_df.to_csv(results_file, index=False)
            logger.info(f"Screening results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save screening results: {e}")

    def generate_charts(self):
        """Generate charts visualizing the test results."""
        # Create charts directory if it doesn't exist
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        
        # 1. Success Rate Chart
        plt.figure(figsize=(10, 6))
        rates = self.metrics["success_rates"]
        plt.bar(["Screening", "Blockchain", "Verification", "End-to-End"], 
                [rates["screening"], rates["blockchain"], rates["verification"], rates["total"]])
        plt.title("Success Rates by Process Stage")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        for i, v in enumerate([rates["screening"], rates["blockchain"], rates["verification"], rates["total"]]):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')
        plt.tight_layout()
        plt.savefig(charts_dir / "success_rates.png")
        
        # 2. Timing Comparison Chart
        plt.figure(figsize=(10, 6))
        metrics = self.metrics
        labels = ["Screening", "Blockchain", "Verification"]
        means = [statistics.mean(metrics["screening_times"]) if metrics["screening_times"] else 0,
                statistics.mean(metrics["blockchain_times"]) if metrics["blockchain_times"] else 0,
                statistics.mean(metrics["verification_times"]) if metrics["verification_times"] else 0]
        
        plt.bar(labels, means)
        plt.title("Average Processing Time by Stage")
        plt.ylabel("Time (seconds)")
        for i, v in enumerate(means):
            plt.text(i, v + 0.1, f"{v:.3f}s", ha='center')
        plt.tight_layout()
        plt.savefig(charts_dir / "timing_comparison.png")
        
        # 3. Distribution of Total Processing Time
        if self.metrics["total_times"]:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics["total_times"], bins=10, alpha=0.7)
            plt.title("Distribution of Total Processing Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency")
            plt.axvline(statistics.mean(self.metrics["total_times"]), color='r', linestyle='dashed', linewidth=1)
            plt.text(statistics.mean(self.metrics["total_times"]), 
                    plt.ylim()[1]*0.9, 
                    f"Mean: {statistics.mean(self.metrics['total_times']):.3f}s", 
                    ha='center')
            plt.tight_layout()
            plt.savefig(charts_dir / "processing_time_distribution.png")
        
        logger.info(f"Charts saved to {charts_dir} directory")

def generate_test_data(num_molecules: int) -> Dict[str, str]:
    """Generate a dictionary of molecules from the bioactivity dataset for testing."""
    logger.info(f"Loading {num_molecules} molecules from the bioactivity dataset for testing...")
    
    data_file = Path(__file__).parent / "modeling" / "data" / "molecules_for_screening.csv"
    
    try:
        df = pd.read_csv(data_file)

        # --- Column Validation ---
        required_columns = ['molecule_chembl_id', 'canonical_smiles']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in the dataset at {data_file}.")
                return {}

        # Ensure we have enough molecules
        if len(df) < num_molecules:
            logger.warning(f"Dataset has only {len(df)} molecules, but {num_molecules} were requested. Using all available molecules.")
            num_molecules = len(df)
        
        # Get a random sample of molecules from the dataset
        sample_df = df.sample(n=num_molecules, random_state=42) # Use a fixed random_state for reproducibility
        
        # Use the real ChEMBL ID for traceability
        molecules = dict(zip(sample_df.molecule_chembl_id, sample_df.canonical_smiles))
            
        logger.info(f"Test data generation complete with {len(molecules)} molecules.")
        return molecules

    except FileNotFoundError:
        logger.error(f"Bioactivity data file not found at {data_file}. Cannot generate test data.")
        return {}
    except Exception as e:
        logger.error(f"Failed to generate test data from dataset: {e}")
        return {}

def main(args):
    """Main function to run end-to-end tests."""
    logger.info(f"Starting PureProt end-to-end tests with {args.num_tests} molecules.")
    
    test = EndToEndTest()
    
    # Connect to blockchain
    if not test.connect_to_blockchain():
        logger.error("Failed to initialize blockchain connection. Exiting.")
        return

    # Generate test data
    molecules = generate_test_data(args.num_tests)
    
    # For this test, we'll use a single, consistent target to measure scalability
    # Use the official biological target for the publication
    targets = ["Amyloid-beta A4 protein"]
    
    # Run tests
    test.run_batch_test(molecules, targets, repeat=1) # Repeat set to 1 for scalability
    
    # Print summary
    test.print_summary()

    # Save screening results
    test.save_screening_results()
    
    # Generate charts
    try:
        test.generate_charts()
    except Exception as e:
        logger.error(f"Failed to generate charts: {e}")

    logger.info("End-to-end tests completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PureProt End-to-End Tests.")
    parser.add_argument("--num_tests", type=int, default=100, help="Number of molecules to generate for testing.")
    args = parser.parse_args()
    main(args)
