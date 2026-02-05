# Experimental Results Summary

**Paper**: Context-Aware Drug Discovery with Zero-Fee Blockchain-Verified Biomaterial Provenance
**Conference**: ICUFN 2026
**Generated**: 2026-02-05T17:42:33.489731

---

## 1. Context-Awareness Validation

**Result**: 6/6 tests passed (100.0%)

**Key Finding**: All 6 context-awareness tests passed. The system correctly produces identical hashes for identical contexts and distinct hashes when any contextual parameter changes. The system captures 9 distinct context fields in each provenance record.

### Context Schema Table

The following fields are captured in each provenance record and used in master hash computation:

| Field | Description | Hash Algorithm | On-Chain |
|-------|-------------|----------------|----------|
| `biomaterial_id` | BioPassport material identifier (e.g., bio:cell_line:hela-001) | N/A (stored directly) | Yes |
| `credential_hash` | SHA-256 hash of biomaterial credentials | SHA-256 | Yes |
| `molecule_id` | Molecule identifier from screening | N/A (stored directly) | Yes |
| `smiles` | SMILES molecular structure string | N/A (stored directly) | Yes* |
| `model_hash` | SHA-256 hash of AI model weights/version | SHA-256 | Yes |
| `parameters_hash` | SHA-256 hash of screening parameters | SHA-256 | Yes |
| `results_hash` | SHA-256 hash of screening results | SHA-256 | Yes |
| `timestamp` | ISO-8601 timestamp of record creation | N/A | Yes |
| `master_hash` | SHA-256 of all above fields (canonical order) | SHA-256 | Yes |

*See privacy note below.

---

## 2. Deterministic Reproducibility

**Result**: 100% hash reproducibility rate

**Key Finding**: Across 40 re-executions (10 runs x 4 test categories), PureProtX achieved a 100% hash reproducibility rate. Identical inputs under identical context consistently produce identical provenance hashes, demonstrating deterministic behavior suitable for regulatory compliance and scientific reproducibility.

---

## 3. Blockchain Verification Performance

### Workflow Definition

**Benchmarked Workflow**: AUDIT PATH
- `verify biomaterial -> create audit record -> anchor hash -> return receipt`
- **Includes**: Credential verification (local), audit record creation (local), blockchain anchoring
- **Excludes**: AI inference, molecular docking (treated as separate workloads)
- **Molecule-to-Transaction Ratio**: 1:1 (no batching)

The case study uses AI inference as the screening workload; docking is treated as an optional module and not included in latency benchmarking.

### Configuration
- Network: PureChain (Chain ID: 900520900520)
- Consensus: PoA (Proof of Authority)
- Gas Cost: **0 PCC (ZERO FEE)**

### Latency Breakdown

**Local Operations (no chain read):**
| Operation | Description | p50 | p95 | p99 |
|-----------|-------------|-----|-----|-----|
| Verification | Local policy evaluation + SHA-256 | 8.3탎 | 14.4탎 | 39.0탎 |
| Audit Creation | JSON serialization + SHA-256 | 31.1탎 | 47.4탎 | 65.0탎 |

**Blockchain Operations:**
| Operation | Description | p50 | p95 | p99 |
|-----------|-------------|-----|-----|-----|
| TX Submit | Sign + RPC send_raw_transaction | 30.7ms | 73.8ms | 529.4ms |
| TX Finality | Block inclusion + receipt | 1981.7ms | 3206.9ms | 3450.1ms |
| Total Anchoring | Submit + Finality | 2063.6ms | 3241.1ms | 3483.4ms |

### Sample Transaction Receipt

| Field | Value |
|-------|-------|
| TX Hash | `55f78133ba2c34c2...` |
| Block Number | 854619 |
| Status | SUCCESS |
| Gas Used | 158561 |
| Submit Latency | 24.67ms |
| Finality Latency | 1650.11ms |
| Total Latency | 1674.78ms |

**Key Finding**: On PureChain (Chain ID: 900520900520), local operations (verification + audit creation) completed in 8탎 + 31탎 (p50). Blockchain anchoring: tx submit p50=31ms, block finality p50=1982ms, total p50=2064ms. Blockchain represents 100.0% of audit path time. Throughput: 0.49 tx/s (1 molecule = 1 tx, no batching). Zero-fee execution confirmed.

---

## 4. Provenance Completeness

**Artifacts Captured**: 7

**Key Finding**: The system captures 7 distinct artifacts in each provenance record, all anchored on-chain via SHA-256 hashing. Each record links biomaterial credentials (from BioPassport) to drug screening results (from PureProtX), creating a complete, independently verifiable audit trail. The master hash enables single-point verification of the entire provenance chain.

### On-Chain vs Off-Chain Storage

**Prototype Configuration** (this implementation):
- Stores molecule identifiers and SMILES on-chain for demonstration
- Full audit record hashes anchored on-chain

**Recommended Production Configuration**:
- Store only content hashes on-chain
- Keep molecular structures and credentials off-chain
- Use content-addressable storage (IPFS/Arweave) for full records

> **Privacy Note**: For privacy and scalability, production deployment should anchor content hashes on-chain and keep molecular structures/credentials off-chain. Our prototype logs identifiers for demonstration purposes. SMILES and biomaterial IDs may be sensitive in real-world applications.

---

## 5. Drug Discovery Case Study

**Molecules Screened**: 5

**Note**: This case study is a **demonstration workload** to validate the pipeline. It does not represent biological insight or drug discovery results. All predictions are tied to model version hash and dataset hash for reproducibility.

**Key Finding**: The case study demonstrates successful execution of a context-aware drug screening pipeline on 5 molecules. All screening results were logged (100%), reproducible (100%), and independently verifiable (100%) via blockchain-anchored provenance records. Biomaterial verification succeeded for 100% of screenings, demonstrating the integration between BioPassport credential verification and PureProtX AI screening within a unified, auditable workflow.

---

## Summary for Reviewers

This system demonstrates:

1. **Context-Awareness**: Different contexts produce different provenance records; identical contexts produce identical hashes (9 context fields captured)
2. **Deterministic Reproducibility**: 100% hash match rate across re-executions
3. **Blockchain Performance**: Zero-fee execution with submit/finality latency breakdown
4. **Provenance Completeness**: Full audit trail with clear on-chain/off-chain separation
5. **Practical Application**: Demonstration workload with verifiable, reproducible results

### Threat Model Note (Zero-Fee)

PureChain uses Proof-of-Authority (PoA) consensus with zero gas fees. Spam resistance relies on:
- Validator trust (permissioned network)
- Rate limiting at RPC layer
- Access control via private key management

This is suitable for consortium/enterprise deployments where validators are known entities.

