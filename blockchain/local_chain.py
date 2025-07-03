#!/usr/bin/env python3
"""Local Blockchain Setup Script

This script sets up a local blockchain for development and testing using Ganache.
It also deploys the DrugScreeningVerifier contract to the local chain.

Usage:
  python local_chain.py

Requires:
  - ganache package installed via pip
  - solcx package for Solidity compilation
"""

import json
import os
import signal
import subprocess
import sys
import time
from web3 import Web3
import solcx
from pathlib import Path

# --- Configuration ---
CHAIN_ID = 1337
PORT = 8545
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_SOL_PATH = PROJECT_ROOT / "blockchain" / "DrugScreeningVerifier.sol"
BUILD_DIR = PROJECT_ROOT / "build" / "contracts"
COMPILED_CONTRACT_PATH = BUILD_DIR / "Purechain.json"
DEPLOYMENT_INFO_PATH = PROJECT_ROOT / "local_deployment_info.json"

# --- Global Variables ---
ganache_process = None

def start_local_chain():
    """Start a local Ganache blockchain."""
    print("Starting local Ethereum blockchain with Ganache...")
    
    # Check if Ganache is installed
    try:
        # Try to get the ganache version to verify it's installed
        version_check = subprocess.run(
            ["ganache", "--version"], 
            capture_output=True, 
            text=True,
            check=False
        )
        
        if version_check.returncode != 0:
            raise FileNotFoundError("Ganache command not found. Please install with 'sudo npm install -g ganache'")
            
        print(f"Ganache found: {version_check.stdout.strip()}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Build Ganache command
    ganache_command = [
        "ganache",  # Use the globally installed ganache executable
        f"--server.port={PORT}",
        "--wallet.deterministic",  # Use deterministic addresses
        "--database.dbPath=./.ganache-db",  # Persist blockchain data
        "--miner.blockTime=0",  # Mine blocks instantly
        "--chain.chainId=1337"  # Set chain ID via chain namespace (Ganache 7.x syntax)
    ]
    
    print(f"Executing: {' '.join(ganache_command)}")
    
    # Create .ganache-db directory if it doesn't exist
    os.makedirs("./.ganache-db", exist_ok=True)
    
    try:
        # Start Ganache process
        process = subprocess.Popen(
            ganache_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor startup
        print("Waiting for Ganache to start...")
        start_time = time.time()
        
        # Wait for Ganache to start and monitor for errors
        timeout = 10  # seconds
        while time.time() - start_time < timeout:
            if process.poll() is not None:  # Process ended prematurely
                stderr = process.stderr.read()
                stdout = process.stdout.read()
                print(f"ERROR: Ganache failed to start.")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                sys.exit(1)
            
            # Try connecting to verify it's running
            try:
                w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{PORT}"))
                if w3.is_connected():
                    break  # Successfully connected
            except Exception:
                pass  # Connection failed, continue waiting
                
            time.sleep(0.5)
        else:  # Timeout reached
            print("ERROR: Timed out waiting for Ganache to start")
            process.terminate()
            sys.exit(1)
            
        print("✓ Local blockchain started successfully")
        print(f"  Chain ID: {CHAIN_ID}")
        print(f"  RPC URL: http://127.0.0.1:{PORT}")
        print("  Available accounts:")
        
        # Print account info
        accounts = w3.eth.accounts
        for i, account in enumerate(accounts[:5]):  # Print first 5 accounts
            balance = w3.from_wei(w3.eth.get_balance(account), 'ether')
            print(f"    [{i}] {account} ({balance} ETH)")
        
        return process, w3
    
    except Exception as e:
        print(f"Error starting local blockchain: {e}")
        sys.exit(1)

def compile_and_save_contract(contract_sol_path, compiled_contract_path):
    """Compile the smart contract and save its ABI and bytecode to a file."""
    print("\nCompiling contract...")
    
    try:
        # Install and set the solc version
        solc_version = '0.8.20'
        try:
            solcx.install_solc(solc_version)
        except Exception:
            # Might already be installed
            pass
        
        solcx.set_solc_version(solc_version)
        
        # Compile the contract
        with open(contract_sol_path, 'r') as f:
            source_code = f.read()
        
        compiled_sol = solcx.compile_source(
            source_code,
            output_values=['abi', 'bin'],
            solc_version=solc_version
        )
        
        # The main contract is DrugScreeningVerifier
        contract_id, contract_interface = compiled_sol.popitem()
        
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']
        
        # Create the build directory if it doesn't exist
        compiled_contract_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the compiled contract artifact
        artifact = {
            'contractName': contract_id.split(':')[-1],
            'abi': abi,
            'bytecode': bytecode,
        }
        
        with open(compiled_contract_path, 'w') as f:
            json.dump(artifact, f, indent=2)
            
        print(f"✓ Contract compiled successfully and saved to {compiled_contract_path}")
        return abi, bytecode
    
    except Exception as e:
        print(f"Error compiling contract: {e}")
        sys.exit(1)

def deploy_contract(w3, abi, bytecode):
    """Deploy the contract to the local blockchain."""
    print("\nDeploying contract to local blockchain...")
    
    try:
        # Use the first account as the deployer
        deployer_address = w3.eth.accounts[0]
        
        # Create contract instance
        Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
        
        # Build deployment transaction
        tx_data = {
            'from': deployer_address,
            'gas': 2000000,
            'gasPrice': w3.to_wei('10', 'gwei'),
            'nonce': w3.eth.get_transaction_count(deployer_address),
        }
        
        # Deploy contract
        tx_hash = Contract.constructor().transact(tx_data)
        
        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress
        
        print(f"✓ Contract deployed successfully at {contract_address}")
        return contract_address
    
    except Exception as e:
        print(f"Error deploying contract: {e}")
        sys.exit(1)

def save_deployment_info(contract_address, abi, deployment_info_path):
    """Save deployment information to a JSON file."""
    deployment_info = {
        "address": contract_address,
        "abi": abi
    }
    
    try:
        with open(deployment_info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"✓ Deployment info saved to {deployment_info_path}")
    except Exception as e:
        print(f"Error saving deployment info: {e}")

def verify_contract(w3, contract_address, abi):
    """Verify the contract deployment."""
    print("\nVerifying contract deployment...")
    
    try:
        # Get contract bytecode
        bytecode = w3.eth.get_code(contract_address)
        if bytecode and bytecode != '0x':
            print("✓ Contract bytecode verified on-chain")
        else:
            print("❌ No bytecode found at contract address")
            return False
        
        # Create contract instance
        contract = w3.eth.contract(address=contract_address, abi=abi)
        
        # Try to call a read function
        owner = contract.functions.owner().call()
        print(f"✓ Contract read operation successful. Owner: {owner}")
        
        # Try to call resultCount
        try:
            count = contract.functions.resultCount().call()
            print(f"✓ Current result count: {count}")
        except Exception as e:
            print(f"Error calling resultCount (may be zero): {e}")
        
        return True
    
    except Exception as e:
        print(f"Error verifying contract: {e}")
        return False

def cleanup(signum=None, frame=None):
    """Cleanup function to stop Ganache on exit."""
    global ganache_process
    if ganache_process and ganache_process.poll() is None:
        print("\nStopping local blockchain...")
        try:
            ganache_process.terminate()
            ganache_process.wait(timeout=5)
            print("✓ Local blockchain stopped successfully")
        except Exception as e:
            print(f"Error stopping blockchain: {e}")
            try:
                ganache_process.kill()
            except:
                pass
    sys.exit(0)

def main():
    """Main function."""
    global ganache_process
    
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("==== Local Blockchain Setup ====")
    
    try:
        # Start the local blockchain
        ganache_process, w3 = start_local_chain()
        
        # Compile the contract
        abi, bytecode = compile_and_save_contract(CONTRACT_SOL_PATH, COMPILED_CONTRACT_PATH)
        
        # Deploy the contract
        contract_address = deploy_contract(w3, abi, bytecode)
        
        # Save deployment info
        save_deployment_info(contract_address, abi, DEPLOYMENT_INFO_PATH)
        
        # Verify the contract
        verify_contract(w3, contract_address, abi)
        
        print("\n==== Setup Complete ====")
        print(f"Local blockchain is running at http://127.0.0.1:{PORT}")
        print(f"Contract deployed at {contract_address}")
        print(f"Deployment info saved to {DEPLOYMENT_INFO_PATH}")
        print("Press Ctrl+C to stop the blockchain.")
        
        # Keep the script running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            cleanup()
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
