"""Smart Contract Deployment Script

This script compiles and deploys the DrugScreeningVerifier.sol smart contract
to the Purechain network.

"""

import json
import getpass
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version, get_installed_solc_versions
import os

# --- Configuration ---
RPC_URL = "http://43.200.53.250:8548"
CHAIN_ID = 900520900520
CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "DrugScreeningVerifier.sol")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deployment_info.json")

def compile_contract(contract_path: str) -> (dict, str):
    """Compile the Solidity contract and return ABI and bytecode."""
    print("Compiling contract...")
    with open(contract_path, 'r') as f:
        source_code = f.read()

    # Install and set the solc version if not already present
    solc_version = '0.8.20'
    try:
        installed_versions = [str(v) for v in get_installed_solc_versions()]
        if solc_version not in installed_versions:
            print(f"solc v{solc_version} not found. Installing...")
            install_solc(solc_version)
        
        set_solc_version(solc_version)
        print(f"✓ Solidity compiler v{solc_version} is ready.")
    except Exception as e:
        print(f"Error setting up Solidity compiler: {e}")
        # Fallback to any available version if the specific one fails
        pass

    compiled_sol = compile_source(source_code, output_values=['abi', 'bin'])
    contract_id, contract_interface = compiled_sol.popitem()
    
    abi = contract_interface['abi']
    bytecode = contract_interface['bin']
    
    print("✓ Contract compiled successfully")
    return abi, bytecode

def deploy_contract(rpc_url: str, chain_id: int, abi: dict, bytecode: str):
    """Deploy the contract to the blockchain."""
    # Connect to the blockchain
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to the blockchain.")

    print(f"Connected to blockchain (Chain ID: {w3.eth.chain_id})")

    # Get private key securely
    private_key = getpass.getpass("Enter your private key to deploy the contract: ")
    try:
        account = w3.eth.account.from_key(private_key)
        print(f"Using account: {account.address}")
    except Exception as e:
        print(f"Invalid private key: {e}")
        return

    # Create contract instance
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build transaction
    print("Building deployment transaction...")
    nonce = w3.eth.get_transaction_count(account.address)
    tx_data = {
        'chainId': chain_id,
        'gas': 2000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': nonce,
    }

    transaction = Contract.constructor().build_transaction(tx_data)

    # Sign and send transaction
    print("Signing and sending transaction...")
    signed_tx = w3.eth.account.sign_transaction(transaction, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    # Wait for transaction receipt
    print(f"Transaction sent with hash: {tx_hash.hex()}")
    print("Waiting for transaction receipt...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    contract_address = tx_receipt.contractAddress
    print(f"\n🎉 Contract deployed successfully at address: {contract_address}")

    return contract_address

def save_deployment_info(contract_address: str, abi: dict, output_path: str):
    """Save the contract address and ABI to a file."""
    deployment_data = {
        "contract_address": contract_address,
        "abi": abi
    }
    with open(output_path, 'w') as f:
        json.dump(deployment_data, f, indent=4)
    print(f"Deployment info saved to {output_path}")


if __name__ == "__main__":
    try:
        # Compile the contract
        contract_abi, contract_bytecode = compile_contract(CONTRACT_PATH)
        
        # Deploy the contract
        deployed_address = deploy_contract(RPC_URL, CHAIN_ID, contract_abi, contract_bytecode)
        
        if deployed_address:
            # Save the deployment information
            save_deployment_info(deployed_address, contract_abi, OUTPUT_PATH)
            
    except Exception as e:
        print(f"\nAn error occurred during deployment: {e}")
