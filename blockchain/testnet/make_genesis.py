#!/usr/bin/env python3
"""
Generate a geth Clique (PoA) genesis for the A4 local testnet.

Mirrors the real PureChain node as observed on the live RPC:
  - client: go-ethereum v1.13.15-stable (Clique PoA)
  - chain id: 900520900520

This is a LOCAL TESTNET replica for protocol-level failure-injection
(Reviewer R1.2). It is NOT PureChain mainnet and never contacts it.

Clique extraData layout:
  32-byte vanity (zero) || sorted signer addresses (20 bytes each) || 65-byte seal (zero)
Signer addresses MUST be ascending (geth rejects unsorted signer lists).

Usage:
  python make_genesis.py 0xADDR1 0xADDR2 0xADDR3 0xADDR4 > genesis.json
"""

import json
import sys

CHAIN_ID = 900520900520          # real PureChain chain id (mirrored on testnet)
CLIQUE_PERIOD = 2                # seconds/block; approximates PureChain's ~2s consensus
CLIQUE_EPOCH = 30000


def build_extradata(addresses):
    signers = sorted(a.lower().replace("0x", "") for a in addresses)
    for s in signers:
        assert len(s) == 40, f"bad address length: {s}"
    vanity = "00" * 32
    seal = "00" * 65
    return "0x" + vanity + "".join(signers) + seal


def build_genesis(addresses):
    alloc = {
        a.lower(): {"balance": "0x200000000000000000000000000000000000000000000000000000000000000"}
        for a in addresses
    }
    return {
        "config": {
            "chainId": CHAIN_ID,
            "homesteadBlock": 0,
            "eip150Block": 0,
            "eip155Block": 0,
            "eip158Block": 0,
            "byzantiumBlock": 0,
            "constantinopleBlock": 0,
            "petersburgBlock": 0,
            "istanbulBlock": 0,
            "berlinBlock": 0,
            # London intentionally NOT enabled: keeps a fixed-zero fee market so
            # zero-gas-price transactions are accepted, matching PureChain's
            # "zero-gas PoA" behaviour.
            "clique": {"period": CLIQUE_PERIOD, "epoch": CLIQUE_EPOCH},
        },
        "nonce": "0x0",
        "timestamp": "0x0",
        "extraData": build_extradata(addresses),
        "gasLimit": "0x1c9c380",   # 30,000,000
        "difficulty": "0x1",
        "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "coinbase": "0x0000000000000000000000000000000000000000",
        "alloc": alloc,
        "number": "0x0",
        "gasUsed": "0x0",
        "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
    }


if __name__ == "__main__":
    addrs = sys.argv[1:]
    if len(addrs) < 1:
        sys.exit("provide signer addresses")
    json.dump(build_genesis(addrs), sys.stdout, indent=2)
