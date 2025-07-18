"""Purechain Blockchain Connector Module

This module provides the PurechainConnector class, which handles all interactions
with the Purechain blockchain, including connecting to the network, loading smart
contracts, and sending transactions to record and verify molecular screening data.
"""
import json
import logging
import os
import time
import hashlib
from pathlib import Path
import threading
from typing import Dict, Any, Tuple

from web3 import Web3
from dotenv import load_dotenv

# Initialize logger for this module
logger = logging.getLogger(__name__)

class PurechainConnector:
    """Handles connection and interaction with the Purechain network."""

    def __init__(self, rpc_url, contract_address, private_key=None, chain_id=1):
        """Initialize the connector.
        
        Args:
            rpc_url: URL of the blockchain node
            contract_address: Address of the deployed contract
            private_key: Private key for signing transactions (optional for local Ganache)
            chain_id: Chain ID of the blockchain (default: 1 for Ethereum mainnet)
        """
        self.logger = logging.getLogger(__name__)
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.private_key = private_key
        self.chain_id = chain_id
        self.w3 = None
        self.contract = None
        self.wallet_address = None
        self.tx_lock = threading.Lock()
        self.dev_mode = (chain_id == 1337) or ('127.0.0.1' in rpc_url) or ('localhost' in rpc_url)
        
        # Set up Web3
        try:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to blockchain at {rpc_url}")
            
            # Handle wallet address based on mode
            if self.dev_mode:
                # In development mode, use the first account from Ganache
                if not self.w3.eth.accounts:
                    raise ValueError("No accounts available in Ganache. Make sure it's running.")
                self.wallet_address = self.w3.eth.accounts[0]
                self.logger.info(f"Development mode: Using Ganache account {self.wallet_address}")
            else:
                # In production mode, require a private key
                if not private_key:
                    raise ValueError("Private key is required for non-development chains")
                account = self.w3.eth.account.from_key(private_key)
                self.wallet_address = account.address
            
            # Load contract ABI from deployment info file
            # Default location is local_deployment_info.json in the project root
            deployment_info_path = Path(__file__).parent.parent / "local_deployment_info.json"
            
            # Try to load ABI from deployment info file
            try:
                with open(deployment_info_path, "r") as f:
                    deployment_info = json.load(f)
                    contract_abi = deployment_info.get("abi")
                    if not contract_abi:
                        raise ValueError(f"No ABI found in {deployment_info_path}")
            except Exception as e:
                self.logger.error(f"Failed to load contract ABI from {deployment_info_path}: {e}")
                raise
            
            self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
            self.logger.info(f"Connected to blockchain at {rpc_url} (Chain ID: {self.chain_id})")
            self.logger.info(f"Using wallet: {self.wallet_address}")
            self.logger.info(f"Contract address: {contract_address}")
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain connection: {e}")
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
        start_time = time.time()
        self.logger.info(f"--- Verifying result for Tx {tx_hash} (Client-Side) ---")
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            func_selector, func_params = self.contract.decode_function_input(tx.input)

            on_chain_hash_bytes = func_params.get('resultHash')

            if on_chain_hash_bytes is None:
                return {"verified": False, "reason": "Hash not found in transaction input.", "duration": time.time() - start_time}

            self.logger.info(f"  - Original Hash: {original_hash_bytes.hex()}")
            self.logger.info(f"  - On-chain Hash: {on_chain_hash_bytes.hex()}")

            verified = original_hash_bytes == on_chain_hash_bytes
            if verified:
                self.logger.info("  - SUCCESS: On-chain hash matches original hash.")
            else:
                self.logger.error("  - FAILURE: On-chain hash does NOT match original hash.")

            return {"verified": verified, "duration": time.time() - start_time}
        except Exception as e:
            self.logger.error(f"Client-side verification failed: {e}", exc_info=True)
            return {"verified": False, "reason": str(e), "duration": time.time() - start_time}

    def record_and_verify_result(self, result_hash_bytes: bytes, molecule_data_hash_bytes: bytes, molecule_id: str) -> Dict[str, Any]:
        """
        Records a screening result on the blockchain using pre-computed hashes.

        This method is thread-safe.
        """
        start_time = time.time()
        self.logger.info("--- Recording Result on Purechain ---")

        with self.tx_lock:
            try:

                nonce = self.w3.eth.get_transaction_count(self.wallet_address)
                tx_params = {
                    'from': self.wallet_address,
                    'nonce': nonce,
                    'gas': 300000,
                    'chainId': self.chain_id,
                }

                if self.chain_id == 900520900520:
                    tx_params['gasPrice'] = 0
                else:
                    tx_params['gasPrice'] = self.w3.to_wei('20', 'gwei')
                
                tx = self.contract.functions.recordScreeningResult(
                    result_hash_bytes, molecule_data_hash_bytes, molecule_id
                ).build_transaction(tx_params)

                if self.dev_mode:
                    tx_hash = self.w3.eth.send_transaction(tx)
                else:
                    signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
                    tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                duration = time.time() - start_time

                if receipt.status != 1:
                    return {"success": False, "error": "Transaction failed on-chain", "duration": duration}

                events = self.contract.events.ResultRecorded().process_receipt(receipt)
                numeric_id = events[0]['args']['numericId'] if events else None

                return {
                    "success": True, 
                    "tx_hash": tx_hash.hex(), 
                    "block_number": receipt.blockNumber, 
                    "gas_used": receipt.gasUsed, 
                    "numeric_id": numeric_id,
                    "duration": duration
                }

            except Exception as e:
                self.logger.error(f"An error occurred while recording the result: {e}", exc_info=True)
                return {"success": False, "error": str(e), "duration": time.time() - start_time}

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
