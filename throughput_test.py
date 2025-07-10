import time
import os
import json
import logging
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from modeling.molecular_modeling import BindingAffinityModel
from blockchain.purechain_connector import PurechainConnector

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Suppress noisy logs from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Initialization ---
try:
    model = BindingAffinityModel()
    
    # Load contract info and initialize connector
    with open('local_deployment_info.json', 'r') as f:
        deployment_info = json.load(f)
        contract_address = deployment_info['address']
        private_key = deployment_info['private_key']

    purechain_connector = PurechainConnector(
        rpc_url="http://127.0.0.1:8545",
        private_key=private_key,
        contract_address=contract_address,
        chain_id=1337
    )
    logger.info("Models and connectors initialized successfully.")
except (FileNotFoundError, KeyError) as e:
    logger.error(f"Failed to load deployment info: {e}. Please run local_chain.py first.", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.error(f"Failed to initialize models or connectors: {e}", exc_info=True)
    sys.exit(1)

# --- Worker Function ---
def process_molecule(molecule_id, smiles):
    """Full pipeline for a single molecule: predict, hash, and record."""
    try:
        # 1. Predict binding affinity
        predicted_affinity = model.predict(smiles)

        # 2. Prepare data for blockchain, matching the structure expected by the connector.
        result_data = {
            "molecule_id": molecule_id,
            "molecule_data": {
                "smiles": smiles,
            },
            "prediction_results": {
                "predicted_binding_affinity": predicted_affinity,
                "model_version": model.get_version(),
            },
            "timestamp": time.time()
        }

        # 3. Record result on the blockchain
        bc_result_data = purechain_connector.record_and_verify_result(result_data)

        # 4. Return the result
        return molecule_id, bc_result_data

    except Exception as e:
        logger.error(f"Error processing molecule {molecule_id}: {e}", exc_info=True)
        return molecule_id, {"success": False, "error": str(e)}

# --- Main Test Function ---
def run_throughput_test(max_workers=10, num_molecules=100):
    """Runs a concurrent test to measure transaction throughput."""
    logger.info(f"Starting throughput test with {max_workers} concurrent workers for {num_molecules} molecules.")

    DATA_FILE = os.path.join('modeling', 'data', 'molecules_for_screening.csv')

    # Load and de-duplicate molecules
    try:
        df = pd.read_csv(DATA_FILE)
        if 'molecule_chembl_id' not in df.columns or 'canonical_smiles' not in df.columns:
            raise KeyError("CSV must contain 'molecule_chembl_id' and 'canonical_smiles' columns.")
        
        df.drop_duplicates(subset=['molecule_chembl_id'], inplace=True)
        df = df.head(num_molecules)
        logger.info(f"Loaded {len(df)} unique molecules for throughput testing.")

    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error loading data: {e}. Please check the file path and format.", exc_info=True)
        return

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_molecule, row['molecule_chembl_id'], row['canonical_smiles']): row['molecule_chembl_id'] for _, row in df.iterrows()}

        for future in as_completed(futures):
            try:
                molecule_id, result_data = future.result()
                if result_data:
                    results.append(result_data)
                    success_status = result_data.get('success', 'Unknown')
                    logger.info(f"Aggregated result for {molecule_id}. Success: {success_status}")
            except Exception as e:
                molecule_id = futures.get(future, 'unknown')
                logger.error(f"Error processing future for molecule {molecule_id}: {e}", exc_info=True)

    end_time = time.time()
    total_time = end_time - start_time
    
    # --- Results Summary ---
    logger.info("--- Throughput Test Summary ---")
    
    successful_transactions = [r for r in results if r and r.get('success')]
    success_count = len(successful_transactions)
    
    logger.info(f"Total molecules submitted for processing: {len(df)}")
    logger.info(f"Total successful transactions: {success_count} / {len(df)}")
    logger.info(f"Total time taken: {total_time:.2f} seconds")

    if total_time > 0 and success_count > 0:
        tps = success_count / total_time
        logger.info(f"Blockchain Throughput: {tps:.2f} TPS")
        
        gas_used_list = [r.get('gas_used', 0) for r in successful_transactions if r.get('gas_used')]
        if gas_used_list:
            avg_gas = sum(gas_used_list) / len(gas_used_list)
            logger.info(f"Average gas used per successful transaction: {avg_gas:,.0f}")
    else:
        logger.info("Not enough successful transactions or time elapsed to calculate TPS.")

    # Save results to a file
    with open('throughput_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Saved detailed results to throughput_results.json")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Example: python throughput_test.py 20 100
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    molecules = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_throughput_test(max_workers=workers, num_molecules=molecules)
