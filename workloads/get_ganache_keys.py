"""
Script to extract Ganache account information for debugging.
"""
import json
import requests
from web3 import Web3

# Connect to Ganache
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
print(f"Connected to Ganache: {w3.is_connected()}")
print(f"Chain ID: {w3.eth.chain_id}")

# Get accounts and balances
print("\nAccounts and balances:")
for i, account in enumerate(w3.eth.accounts):
    balance = w3.eth.get_balance(account)
    print(f"Account {i}: {account} - {balance / 1e18} ETH")

# Get private keys from Ganache directly (using its internal RPC method)
try:
    # This is a special non-standard Ganache method
    response = requests.post(
        "http://127.0.0.1:8545",
        json={"jsonrpc": "2.0", "method": "eth_accounts", "params": [], "id": 1}
    )
    accounts = response.json()["result"]
    
    # Get the private keys
    response = requests.post(
        "http://127.0.0.1:8545",
        json={"jsonrpc": "2.0", "method": "evm_snapshot", "params": [], "id": 1}
    )
    
    print("\nTrying to get private keys (may not work with all Ganache versions):")
    for i, account in enumerate(accounts):
        if i == 0:
            print(f"Account {i}: {account}")
            print("This is the address we should use for deployment.")
            print("Try checking the Ganache UI for the corresponding private key.")
            print("Or use the mnemonic to regenerate all keys.")
except Exception as e:
    print(f"Could not get private keys directly: {e}")
    print("Please check the Ganache UI for private keys.")

print("\nIMPORTANT: If using ganache-cli or ganache core, your first account's private key is needed.")
print("You'll need to copy this key from the Ganache UI or console output.")
