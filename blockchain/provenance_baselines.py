#!/usr/bin/env python3
"""
PureProtX Provenance Baselines (Reviewer Comment R2.5)

Reviewer 2 correctly noted that comparing PoA-squared anchoring only against
"N individual commits vs 1 Merkle batch" is a strawman (the batch win is a
trivial O(N) -> O(1) accounting result). This module implements *legitimate*
alternative provenance mechanisms so PoA-squared can be compared honestly and
the axes on which it actually wins (or loses) are made explicit.

Mechanisms implemented here (all local, no network, no mainnet):

  1. Ed25519 signed append-only hash chain
     A certificate-transparency-style log: each entry links the previous
     entry's hash and is signed by a single Ed25519 key. Tamper-evident, but
     trust reduces to a single key holder (whoever holds the private key can
     rewrite and re-sign history). No Byzantine fault tolerance.

  2. IPFS-style content addressing (LOCAL CID computation only)
     Computes a CIDv1 (raw codec, sha2-256 multihash, base32 multibase) over
     the canonical payload bytes. This is exactly the CID that
     `ipfs add --cid-version 1 --raw-leaves` produces for a single-block
     (<256 KiB) file. IMPORTANT HONESTY NOTE: default `ipfs add` (no flags)
     wraps bytes in a UnixFS/dag-pb node and yields a *different* CIDv0
     (Qm...). We compute the raw-leaf CID here; we measure the
     content-addressing PROPERTY (immutability + hash-keyed verification),
     NOT network storage, pinning, retrieval, or availability. Those are not
     measured and must not be claimed.

  3. Merkle batch root (re-implemented to mirror
     experiments/scalability_benchmark.py so A3 stays dependency-light)
     One anchor commits N record hashes; single-record verification needs a
     log2(N) inclusion proof.

The PoA-squared / PureChain figures are NOT re-measured here (that would touch
live mainnet). They are reused from the already-committed on-chain results in
IOT_EXPERIMENT_RESULTS.md (V-C consensus latency, V-E scalability) by the
companion benchmark script.

Standalone:
  python -m blockchain.provenance_baselines
"""

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

GENESIS_PREV_HASH = "0" * 64


# ------------------------------------------------------------------
# Hashing / canonicalisation helpers
# ------------------------------------------------------------------

def _sha256(data: bytes) -> bytes:
    """Return raw 32-byte SHA-256 digest."""
    return hashlib.sha256(data).digest()


def _sha256_hex(data: bytes) -> str:
    """Return hex SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """
    Deterministic canonical JSON encoding (sorted keys, no whitespace).

    Mirrors the canonicalisation rule used elsewhere in PureProtX so the
    baselines hash payloads the same way the main pipeline does.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------------------
# Mechanism 1: Ed25519 signed append-only hash chain
# ------------------------------------------------------------------

@dataclass
class SignedLogEntry:
    """A single append-only log entry (hash-chained + Ed25519 signed)."""
    index: int
    prev_hash: str            # hex sha256 of the previous entry (genesis = 0*64)
    payload_hash: str         # hex sha256 of the canonical payload
    entry_hash: str           # hex sha256 of the canonical entry (excl. signature)
    signature: str            # hex Ed25519 signature over entry_hash bytes


class Ed25519SignedLog:
    """
    Append-only, hash-chained, single-signer provenance log.

    Trust model: CENTRALISED. A single Ed25519 key authenticates every entry.
    Anyone holding the private key can rewrite the entire history and re-sign
    it; immutability therefore depends on external publication / trusting the
    key holder, not on the structure itself. No BFT. This is the honest
    characterisation used in the comparison table.
    """

    def __init__(self, private_key: Optional[Ed25519PrivateKey] = None):
        self._sk = private_key or Ed25519PrivateKey.generate()
        self._pk: Ed25519PublicKey = self._sk.public_key()
        self.entries: List[SignedLogEntry] = []

    def _core_bytes(self, index: int, prev_hash: str, payload_hash: str) -> bytes:
        return canonical_json(
            {"index": index, "prev_hash": prev_hash, "payload_hash": payload_hash}
        )

    def append(self, payload: Any) -> SignedLogEntry:
        """Append a payload; returns the created (signed, chained) entry."""
        index = len(self.entries)
        prev_hash = self.entries[-1].entry_hash if self.entries else GENESIS_PREV_HASH
        payload_hash = _sha256_hex(canonical_json(payload))
        entry_hash_bytes = _sha256(self._core_bytes(index, prev_hash, payload_hash))
        signature = self._sk.sign(entry_hash_bytes)
        entry = SignedLogEntry(
            index=index,
            prev_hash=prev_hash,
            payload_hash=payload_hash,
            entry_hash=entry_hash_bytes.hex(),
            signature=signature.hex(),
        )
        self.entries.append(entry)
        return entry

    def verify(self, public_key: Optional[Ed25519PublicKey] = None) -> bool:
        """
        Verify the full log: chain linkage + per-entry Ed25519 signature.

        Returns True only if every link and signature is valid.
        """
        pk = public_key or self._pk
        expected_prev = GENESIS_PREV_HASH
        for i, e in enumerate(self.entries):
            if e.index != i or e.prev_hash != expected_prev:
                return False
            recomputed = _sha256(
                self._core_bytes(e.index, e.prev_hash, e.payload_hash)
            )
            if recomputed.hex() != e.entry_hash:
                return False
            try:
                pk.verify(bytes.fromhex(e.signature), recomputed)
            except InvalidSignature:
                return False
            expected_prev = e.entry_hash
        return True

    def public_key(self) -> Ed25519PublicKey:
        return self._pk


# ------------------------------------------------------------------
# Mechanism 2: IPFS-style content addressing (local CIDv1 raw)
# ------------------------------------------------------------------

def compute_cid_v1_raw(data: bytes) -> str:
    """
    Compute a CIDv1 (raw codec, sha2-256, base32 multibase) over `data`.

    Layout:  <version:0x01> <codec:0x55 raw> <multihash>
             multihash = <fn:0x12 sha2-256> <len:0x20 = 32> <digest>
    Multibase: base32 lower, RFC4648, no padding, prefix 'b'.

    This equals `ipfs add --cid-version 1 --raw-leaves <file>` for a single
    block (<256 KiB). It does NOT equal default `ipfs add` (dag-pb CIDv0).
    Measures the content-addressing property only; no network involved.
    """
    digest = hashlib.sha256(data).digest()
    multihash = bytes([0x12, 0x20]) + digest          # sha2-256, 32-byte length
    cid_bytes = bytes([0x01, 0x55]) + multihash        # CIDv1, raw codec
    b32 = base64.b32encode(cid_bytes).decode("ascii").lower().rstrip("=")
    return "b" + b32


def verify_cid_v1_raw(data: bytes, cid: str) -> bool:
    """Re-derive the CID from data and compare (content-address verification)."""
    return compute_cid_v1_raw(data) == cid


# ------------------------------------------------------------------
# Mechanism 3: Merkle batch (mirrors experiments/scalability_benchmark.py)
# ------------------------------------------------------------------

def compute_merkle_root(hashes: List[bytes]) -> bytes:
    """
    Bitcoin-style Merkle root over 32-byte SHA-256 leaves.
    Duplicates the last element on odd layers.
    """
    if not hashes:
        return _sha256(b"")
    layer = list(hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [_sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def merkle_proof(hashes: List[bytes], index: int) -> List[Tuple[str, bytes]]:
    """
    Inclusion proof for leaf `index`: list of (side, sibling_hash) from leaf
    to root. `side` is 'L' or 'R' indicating which side the sibling sits on.
    """
    proof: List[Tuple[str, bytes]] = []
    layer = list(hashes)
    idx = index
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        sibling = idx ^ 1
        side = "R" if sibling > idx else "L"
        proof.append((side, layer[sibling]))
        layer = [_sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: bytes, proof: List[Tuple[str, bytes]], root: bytes) -> bool:
    """Verify a Merkle inclusion proof recomputes the given root."""
    h = leaf
    for side, sibling in proof:
        h = _sha256(sibling + h) if side == "L" else _sha256(h + sibling)
    return h == root


# ------------------------------------------------------------------
# Self-test / demonstration
# ------------------------------------------------------------------

def _selftest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Ed25519 signed log: append 5, verify, then tamper and confirm reject.
    log = Ed25519SignedLog()
    for i in range(5):
        log.append({"record": i, "score": 6.7 + i * 0.01})
    assert log.verify(), "fresh log should verify"
    # Tamper with a payload hash and confirm the log now fails.
    log.entries[2].payload_hash = _sha256_hex(b"tampered")
    assert not log.verify(), "tampered log must fail verification"
    logger.info("[1] Ed25519 signed log: verify PASS, tamper REJECTED  OK")

    # 2. Content addressing: CID is deterministic; a 1-byte change diverges.
    data = canonical_json({"target": "CHEMBL243", "pred": 6.7739})
    cid = compute_cid_v1_raw(data)
    assert verify_cid_v1_raw(data, cid)
    data2 = canonical_json({"target": "CHEMBL243", "pred": 6.7749})
    assert compute_cid_v1_raw(data2) != cid
    logger.info("[2] CIDv1 raw: %s  (deterministic, tamper-divergent)  OK", cid)

    # 3. Merkle proof round-trips.
    leaves = [_sha256(f"rec_{i}".encode()) for i in range(8)]
    root = compute_merkle_root(leaves)
    for i in range(8):
        assert verify_merkle_proof(leaves[i], merkle_proof(leaves, i), root)
    # A wrong leaf must fail.
    assert not verify_merkle_proof(_sha256(b"fake"), merkle_proof(leaves, 0), root)
    logger.info("[3] Merkle proofs: 8/8 inclusion PASS, forgery REJECTED  OK")

    logger.info("\nAll provenance-baseline self-tests passed.")


if __name__ == "__main__":
    _selftest()
