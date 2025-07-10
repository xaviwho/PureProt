"""Smart Contract Deployment Script

This script compiles and deploys the DrugScreeningVerifier.sol smart contract.
It is designed to be run from the command line and supports different blockchain environments.

"""

import json
import argparse
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version, get_installed_solc_versions
import os

# --- Configuration ---
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
        pass

    compiled_sol = compile_source(source_code, output_values=['abi', 'bin'])
    contract_id, contract_interface = compiled_sol.popitem()
    
    abi = contract_interface['abi']
    bytecode = contract_interface['bin']
    
    return abi, bytecode

def deploy_contract(rpc_url: str, chain_id: int, private_key: str = None, abi: dict = None, bytecode: str = None, dev_mode: bool = False):
    """Deploy the contract to the blockchain.
    
    Args:
        rpc_url: URL of the blockchain node
        chain_id: Chain ID of the blockchain
        private_key: Private key for signing transactions (optional if dev_mode is True)
        abi: Contract ABI (required)
        bytecode: Contract bytecode (required)
        dev_mode: If True, will use the first account from the node (useful for Ganache)
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to the blockchain at {rpc_url}.")

    print(f"Connected to blockchain (Chain ID: {w3.eth.chain_id})")
    
    # Handle development mode (Ganache)
    if chain_id == 1337 or dev_mode:
        print("Development mode detected. Using first account from node...")
        if not w3.eth.accounts:
            print("No accounts found in the node. Make sure Ganache is running.")
            return None
            
        # Use the first account from the node (Ganache keeps these unlocked)
        account_address = w3.eth.accounts[0]
        print(f"Using first Ganache account: {account_address}")
    else:
        # For production, require private key
        if not private_key:
            print("Private key is required for non-development chains")
            return None
            
        try:
            account = w3.eth.account.from_key(private_key)
            account_address = account.address
            print(f"Using account: {account_address}")
        except Exception as e:
            print(f"Invalid private key: {e}")
            return None

    # Create contract instance
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    print("Building deployment transaction...")
    nonce = w3.eth.get_transaction_count(account_address)
    
    # Build transaction
    deployment_tx = Contract.constructor().build_transaction({
        'from': account_address,
        'nonce': nonce,
        'gas': 3000000,  # Generous gas limit for deployment
        'gasPrice': w3.to_wei('20', 'gwei'),  # Standard gas price for local testnet
        'chainId': chain_id
    })

    print("Signing and sending transaction...")
    
    # Handle different transaction signing methods based on mode
    if chain_id == 1337 or dev_mode:
        # In development mode with Ganache, we can send the transaction directly
        # as Ganache automatically signs transactions from unlocked accounts
        try:
            tx_hash = w3.eth.send_transaction(deployment_tx)
            print(f"Transaction sent with hash: {tx_hash.hex()}")
        except Exception as e:
            print(f"Failed to send transaction: {e}")
            return None
    else:
        # In production mode, we need to sign with the private key
        try:
            signed_tx = w3.eth.account.sign_transaction(deployment_tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"Transaction sent with hash: {tx_hash.hex()}")
        except Exception as e:
            print(f"Failed to sign or send transaction: {e}")
            return None
    
    print("Waiting for transaction receipt...")
    try:
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    except Exception as e:
        print(f"Failed to get transaction receipt: {e}")
        return None

    contract_address = tx_receipt.contractAddress
    print(f"\nSUCCESS! Contract deployed successfully at address: {contract_address}")

    return contract_address

def save_deployment_info(contract_address: str, abi: dict, output_path: str):
    """Save the contract address and ABI to a file."""
    deployment_data = {
        'address': contract_address,
        'abi': abi
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(deployment_data, f, indent=2)
    print(f"Deployment info saved to {output_path}")

def main():
    """Main function to parse arguments and drive deployment."""
    parser = argparse.ArgumentParser(description="Compile and deploy the DrugScreeningVerifier smart contract.")
    parser.add_argument('--rpc-url', required=True, help='URL of the Ethereum RPC endpoint.')
    parser.add_argument('--chain-id', required=True, type=int, help='Chain ID of the target network.')
    parser.add_argument('--private-key', required=True, help='Private key of the deploying account.')
    parser.add_argument('--output-path', required=True, help='Path to save the deployment_info.json file.')
    args = parser.parse_args()

    contract_path = os.path.join(os.path.dirname(__file__), "DrugScreeningVerifier.sol")

    try:
        abi, bytecode = compile_contract(contract_path)
        contract_address = deploy_contract(
            rpc_url=args.rpc_url,
            chain_id=args.chain_id,
            private_key=args.private_key,
            abi=abi,
            bytecode=bytecode
        )
        if contract_address:
            save_deployment_info(contract_address, abi, args.output_path)
    except Exception as e:
        print(f"\nAn error occurred during deployment: {e}")
