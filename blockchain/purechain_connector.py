"""Purechain Blockchain Connector Module

This module provides the PurechainConnector class, which handles all interactions
with the Purechain blockchain, including connecting to the network, loading smart
contracts, and sending transactions to record and verify molecular screening data.
"""
import json
import logging
import os
import hashlib
from pathlib import Path
from typing import Dict, Any

from web3 import Web3
from dotenv import load_dotenv

# Initialize logger for this module
logger = logging.getLogger(__name__)

class PurechainConnector:
    """Handles connection and interaction with the Purechain network."""

    def __init__(self, rpc_url: str, private_key: str, contract_address: str, chain_id: int):
        """
        Initializes the connector, connects to the blockchain, and loads the contract.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing PurechainConnector...")

        try:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to the blockchain RPC.")
            self.logger.info(f"Successfully connected to blockchain node at {rpc_url}")

            self.private_key = private_key
            self.wallet_address = self.w3.eth.account.from_key(private_key).address
            self.chain_id = chain_id
            self.logger.info(f"Wallet address {self.wallet_address} loaded for chain ID {self.chain_id}.")

            self.contract_address = contract_address
            contract_abi = self.load_contract_abi()
            self.contract = self.w3.eth.contract(address=self.contract_address, abi=contract_abi)
            self.logger.info(f"Smart contract loaded at address: {self.contract_address}")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    @staticmethod
    def load_contract_abi() -> list:
        """Loads the contract ABI from the build file using a robust path."""
        # Construct a robust path to the ABI file relative to this script's location
        project_root = Path(__file__).resolve().parent.parent
        abi_path = project_root / 'build' / 'contracts' / 'Purechain.json'
        
        logger.info(f"Attempting to load contract ABI from: {abi_path}")
        try:
            with open(abi_path, 'r', encoding='utf-8') as f:
                contract_json = json.load(f)
                return contract_json['abi']
        except FileNotFoundError:
            logger.error(f"ABI file not found at {abi_path}. Make sure contracts are compiled and in the correct location.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from ABI file at {abi_path}.")
            raise

    def verify_result_client_side(self, tx_hash: str, original_hash_bytes: bytes) -> Dict[str, Any]:
        """
        Verifies a transaction by decoding its input data, as direct state reads are unavailable.
        """
        self.logger.info(f"--- Verifying result for Tx {tx_hash} (Client-Side) ---")
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            # The first 4 bytes are the function selector
            # The arguments are encoded in the rest of the input data
            func_selector, func_params = self.contract.decode_function_input(tx.input)

            # The hash is the second argument ('resultHash') in the recordResult function
            on_chain_hash_bytes = func_params.get('resultHash')

            if on_chain_hash_bytes is None:
                self.logger.warning("Could not find 'resultHash' in decoded transaction input.")
                return {"verified": False, "reason": "Hash not found in transaction input."}

            self.logger.info(f"  - Original Hash: {original_hash_bytes.hex()}")
            self.logger.info(f"  - On-chain Hash: {on_chain_hash_bytes.hex()}")

            verified = original_hash_bytes == on_chain_hash_bytes
            if verified:
                self.logger.info("  - SUCCESS: On-chain hash matches original hash.")
            else:
                self.logger.error("  - FAILURE: On-chain hash does NOT match original hash.")

            return {"verified": verified}
        except Exception as e:
            self.logger.error(f"Client-side verification failed: {e}", exc_info=True)
            return {"verified": False, "reason": str(e)}

    def record_and_verify_result(self, screening_result: Dict[str, Any]) -> Dict[str, Any]:
        """Records a screening result on the blockchain and verifies it."""
        self.logger.info("--- Recording Result on Purechain ---")
        try:
            # 1. Prepare data and generate hash
            result_json = json.dumps(screening_result, sort_keys=True)
            result_hash_bytes = hashlib.sha256(result_json.encode()).digest()
            self.logger.info(f"  - Generated result hash: {result_hash_bytes.hex()}")

            # 2. Prepare transaction data
            molecule_data_hash_bytes = hashlib.sha256(json.dumps(screening_result['molecule_data'], sort_keys=True).encode()).digest()
            numeric_id = self.contract.functions.resultCount().call()
            self.logger.info(f"  - Current result count: {numeric_id}. New result will have this ID.")

            # 3. Build and send the transaction
            self.logger.info("  - Building transaction...")
            nonce = self.w3.eth.get_transaction_count(self.wallet_address)
            tx_params = {
                'from': self.wallet_address,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('50', 'gwei'),
                'chainId': self.chain_id
            }
            
            tx = self.contract.functions.recordScreeningResult(
                result_hash_bytes,
                molecule_data_hash_bytes,
                screening_result['molecule_id']
            ).build_transaction(tx_params)

            self.logger.info("  - Signing transaction...")
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)

            self.logger.info("  - Sending transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            self.logger.info(f"  - Transaction hash: {tx_hash.hex()}")

            self.logger.info("  - Waiting for transaction receipt...")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            self.logger.info(f"  - Success! Transaction mined in block: {receipt.blockNumber}")
            self.logger.info(f"  - Gas used: {receipt.gasUsed}")
            self.logger.info(f"  - Status: {'Success' if receipt.status == 1 else 'Failed'}")

            if receipt.status != 1:
                self.logger.error("  - Transaction failed to execute.")
                return {"success": False, "error": "Transaction failed"}

            self.logger.info(f"  - Successfully recorded result with numeric ID: {numeric_id}")

            # 4. Client-side verification
            verification_result = self.verify_result_client_side(tx_hash.hex(), result_hash_bytes)

            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "numeric_id": numeric_id,
                "verified": verification_result["verified"],
                "verification_method": "client-side"
            }

        except Exception as e:
            self.logger.error(f"An error occurred while recording the result: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

def main():
    """Main function for standalone testing of the connector."""
    # Basic logging setup for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load environment variables
    load_dotenv()
    rpc_url = os.getenv("GANACHE_RPC_URL", "http://127.0.0.1:8545")
    private_key = os.getenv("TEST_PRIVATE_KEY")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    
    if not all([private_key, contract_address]):
        logger.error("Missing required environment variables: TEST_PRIVATE_KEY, CONTRACT_ADDRESS")
        return

    try:
        # Use Ganache's default chain ID for local testing
        connector = PurechainConnector(rpc_url, private_key, contract_address, chain_id=1337)
        logger.info("PurechainConnector initialized successfully for standalone test.")
        
        # Example screening result for testing
        test_result = {
            "molecule_id": "TEST_MOL_001",
            "molecule_data": {"smiles": "CCO", "molecular_weight": 46.07},
            "results": {"toxicity": 0.1, "binding_affinity": -5.4}
        }
        
        record_response = connector.record_and_verify_result(test_result)
        logger.info(f"Recording response: {json.dumps(record_response, indent=2)}")

    except Exception as e:
        logger.error(f"An error occurred during the standalone test: {e}", exc_info=True)

if __name__ == "__main__":
    main()
