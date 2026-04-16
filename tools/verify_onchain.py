#!/usr/bin/env python3
"""
Independent PureChain Transaction Verifier

Fetches a transaction from the PureChain RPC, decodes the resultHash
field from the contract call input data, and compares it to a locally
provided SHA-256 digest. Prints VERIFIED or MISMATCH.

Usage:
  python tools/verify_onchain.py --tx 0x04d55c... --expected-hash abc123...
  python tools/verify_onchain.py --tx 0x04d55c... --local-json results/some_file.json

Dependencies: web3 (pip install web3)
No private key required -- read-only verification.
"""

import argparse
import hashlib
import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PURECHAIN_RPC = "https://purechainnode.com"
CHAIN_ID = 900520900520
CONTRACT = "0xb8eb74663c1297825b188D8454a469d02Cc7d56C"


def get_web3():
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
    w3 = Web3(Web3.HTTPProvider(PURECHAIN_RPC))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {PURECHAIN_RPC}")
        sys.exit(1)
    return w3


def load_contract_abi():
    """Load ABI from local_deployment_info.json."""
    info_path = os.path.join(PROJECT_ROOT, "local_deployment_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f).get("abi", [])
    return []


def fetch_and_decode_tx(tx_hash: str) -> dict:
    """Fetch a transaction and decode its resultHash from the input data."""
    w3 = get_web3()

    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    tx = w3.eth.get_transaction(tx_hash)
    receipt = w3.eth.get_transaction_receipt(tx_hash)

    info = {
        "tx_hash": tx_hash,
        "block_number": receipt.blockNumber,
        "from": tx["from"],
        "to": tx["to"],
        "status": "SUCCESS" if receipt.status == 1 else "FAILED",
        "gas_used": receipt.gasUsed,
    }

    # Decode the function input using the contract ABI
    abi = load_contract_abi()
    if abi:
        contract = w3.eth.contract(address=CONTRACT, abi=abi)
        try:
            func, params = contract.decode_function_input(tx.input)
            result_hash_bytes = params.get("resultHash", b"")
            molecule_data_hash = params.get("moleculeDataHash", b"")
            molecule_id = params.get("moleculeId", "")

            info["function"] = func.fn_name
            info["resultHash"] = result_hash_bytes.hex() if result_hash_bytes else ""
            info["moleculeDataHash"] = molecule_data_hash.hex() if molecule_data_hash else ""
            info["moleculeId"] = molecule_id
        except Exception as e:
            info["decode_error"] = str(e)
    else:
        # Fallback: raw input data parsing
        # recordScreeningResult(bytes32, bytes32, string)
        # Selector = first 4 bytes, then 32 + 32 + dynamic string
        input_hex = tx.input.hex()
        if len(input_hex) >= 8 + 128:  # selector + 2x bytes32
            info["resultHash"] = input_hex[8:72]  # first bytes32 param
            info["moleculeDataHash"] = input_hex[72:136]  # second bytes32 param
        info["note"] = "Decoded from raw input (no ABI file found)"

    return info


def compute_local_hash(json_path: str) -> str:
    """Compute SHA-256 of a local JSON file (canonical form)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify a PureChain transaction against a local hash"
    )
    parser.add_argument("--tx", required=True, help="Transaction hash (hex)")
    parser.add_argument("--expected-hash", help="Expected SHA-256 hex digest")
    parser.add_argument("--local-json", help="Path to local JSON file to hash")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.expected_hash and not args.local_json:
        # Just fetch and display the transaction
        print(f"Fetching transaction {args.tx}...")
        info = fetch_and_decode_tx(args.tx)
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    # Fetch on-chain hash
    print(f"Fetching transaction {args.tx[:20]}...")
    info = fetch_and_decode_tx(args.tx)
    onchain_hash = info.get("resultHash", "")

    if args.verbose:
        for k, v in info.items():
            print(f"  {k}: {v}")

    # Compute or use expected hash
    if args.local_json:
        expected = compute_local_hash(args.local_json)
        print(f"Local JSON:    {args.local_json}")
    else:
        expected = args.expected_hash

    print(f"On-chain hash: {onchain_hash}")
    print(f"Expected hash: {expected}")

    if onchain_hash == expected:
        print(f"\nResult: VERIFIED")
        print(f"  Block: {info.get('block_number')}")
        print(f"  Molecule: {info.get('moleculeId', 'N/A')}")
    else:
        print(f"\nResult: MISMATCH")
        # Show first difference position
        for i, (a, b) in enumerate(zip(onchain_hash, expected)):
            if a != b:
                print(f"  First difference at position {i}: on-chain='{a}' expected='{b}'")
                break


if __name__ == "__main__":
    main()
