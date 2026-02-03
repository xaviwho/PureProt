# Methodology and System Design

**Paper**: Context-Aware Drug Discovery with Zero-Fee Blockchain-Verified Biomaterial Provenance

**Conference**: ICUFN 2026

---

## 1. Introduction

This document describes the methodology and system design for a context-aware drug discovery platform that integrates biomaterial provenance verification with AI-based virtual screening, anchored on a zero-fee blockchain for complete reproducibility and regulatory compliance.

---

## 2. System Architecture

### 2.1 Overview

The system comprises three integrated layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │  BioPassport  │    │   PureProtX   │    │   Unified     │   │
│  │  (Credentials)│───▶│ (AI Screening)│───▶│ Audit Record  │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    VERIFICATION LAYER                            │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │   Credential  │    │  Deterministic│    │    Master     │   │
│  │  Verification │    │    Hashing    │    │    Hash       │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    BLOCKCHAIN LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PureChain (Zero-Fee Blockchain)             │   │
│  │         Chain ID: 900520900520 | Consensus: PoA          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Description

| Component | Function | Technology |
|-----------|----------|------------|
| BioPassport | Biomaterial credential verification | Smart contracts, Web3 |
| PureProtX | Consensus AI drug screening | SVR, Random Forest, Gradient Boosting |
| Unified Audit Record | Provenance linking | SHA-256, JSON canonicalization |
| PureChain | Immutable audit trail | EVM-compatible, PoA consensus |

---

## 3. BioPassport Integration

### 3.1 Credential Types

The BioPassport system supports four credential types for biomaterial provenance:

| Credential Type | Purpose | Example |
|----------------|---------|---------|
| `IDENTITY` | Cell line STR profile or plasmid sequence fingerprint | HeLa cell authentication |
| `QC_MYCO` | Mycoplasma test result | Negative test certificate |
| `TRANSFER` | Chain-of-custody event | Lab-to-lab transfer record |
| `USAGE_RIGHTS` | MTA restrictions and expiration | Material transfer agreement |

### 3.2 Verification Protocol

A biomaterial is considered **VALID** for drug screening only if ALL conditions are met:

```
VALID = has_IDENTITY ∧ has_QC_MYCO ∧ ¬QUARANTINED ∧ ¬REVOKED ∧ transfer_chain_valid
```

**Algorithm 1: Biomaterial Verification**
```
function VerifyBiomaterial(material_id, required_credentials, at_time):
    policy_checks = {}

    // Query on-chain credential state
    material_data = blockchain.getMaterialCredentials(material_id)

    if material_data is NULL:
        return FAIL("Material not registered")

    // Check each required credential type
    for cred_type in required_credentials:
        cred = material_data.getCredential(cred_type)
        if cred is NULL or cred.expiry < at_time:
            policy_checks[cred_type] = FAIL
        else:
            policy_checks[cred_type] = PASS

    // Check material status
    if material_data.status in [QUARANTINED, REVOKED]:
        return FAIL("Material status invalid")

    // Validate transfer chain
    if not ValidateTransferChain(material_data.transfers):
        return FAIL("Transfer chain has gaps")

    // All checks passed
    credential_hash = SHA256(canonical_json(material_data))
    return PASS(credential_hash, policy_checks)
```

### 3.3 Dual-Layer Verification

The system implements dual-layer verification:

1. **On-Chain Layer**: Policy compliance checks (credential existence, expiration, status)
2. **Off-Chain Layer**: Artifact integrity verification (document hashes, signatures)

---

## 4. Consensus AI Screening

### 4.1 Ensemble Architecture

The AI screening module uses a consensus ensemble of three models:

```
                    ┌─────────────────┐
                    │  Molecular      │
                    │  Features       │
                    │  (2048-bit FP + │
                    │   10 descriptors)│
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │   SVR    │      │  Random  │      │ Gradient │
    │  (RBF)   │      │  Forest  │      │ Boosting │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
         └────────────┬────┴─────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │   Consensus  │
               │   (Average)  │
               └──────────────┘
```

### 4.2 Feature Engineering

**Molecular Descriptors** (10 features):
1. Molecular Weight (MW)
2. LogP (octanol-water partition coefficient)
3. Number of H-bond Donors (HBD)
4. Number of H-bond Acceptors (HBA)
5. Topological Polar Surface Area (TPSA)
6. Number of Rotatable Bonds (RotB)
7. Number of Aromatic Rings (ArRings)
8. Fraction of sp3 Carbons (Fsp3)
9. Number of Heavy Atoms (HeavyAtoms)
10. Formal Charge

**Morgan Fingerprints**: 2048-bit circular fingerprints (radius=2)

### 4.3 Consensus Prediction

The final prediction is the arithmetic mean of individual model predictions:

```
pIC50_consensus = (pIC50_SVR + pIC50_RF + pIC50_GB) / 3
```

This ensemble approach reduces variance and provides more robust predictions than any single model.

---

## 5. Unified Audit Record

### 5.1 Schema Definition

The Unified Audit Record links biomaterial provenance to screening results:

```json
{
  "record_id": "string (16-char hex)",
  "timestamp": "ISO 8601 datetime",

  "biomaterial_provenance": {
    "material_id": "bio:cell_line:<uuid>",
    "credential_hash": "sha256:...",
    "verification_status": "PASS|FAIL",
    "policy_checks": {
      "has_identity": true,
      "has_qc_myco": true,
      "not_quarantined": true,
      "not_revoked": true,
      "transfer_chain_valid": true
    }
  },

  "drug_screening": {
    "molecule_id": "string",
    "smiles": "canonical SMILES",
    "model_hash": "sha256:...",
    "parameters_hash": "sha256:...",
    "results": {
      "consensus_pic50": 6.54,
      "individual_predictions": {
        "svr": 6.4,
        "random_forest": 6.6,
        "gradient_boosting": 6.5
      }
    }
  },

  "master_hash": "sha256:...",
  "blockchain_tx": "0x..."
}
```

### 5.2 Hash Computation

**Algorithm 2: Master Hash Computation**
```
function ComputeMasterHash(record):
    // Create canonical JSON (sorted keys, minimal whitespace)
    components = {
        "record_id": record.record_id,
        "timestamp": record.timestamp,
        "biomaterial_id": record.biomaterial_provenance.material_id,
        "biomaterial_credential_hash": record.biomaterial_provenance.credential_hash,
        "verification_status": record.biomaterial_provenance.verification_status,
        "molecule_id": record.drug_screening.molecule_id,
        "smiles": record.drug_screening.smiles,
        "model_hash": record.drug_screening.model_hash,
        "parameters_hash": record.drug_screening.parameters_hash,
        "screening_results": record.drug_screening.results
    }

    canonical_json = JSON.stringify(components, sort_keys=true, separators=(',', ':'))
    master_hash = SHA256(canonical_json)

    return master_hash
```

### 5.3 Context Fields

The following context fields are captured in each provenance record:

| Field | Description | Affects Hash |
|-------|-------------|--------------|
| biomaterial_id | BioPassport material identifier | Yes |
| biomaterial_credential_hash | Hash of verified credentials | Yes |
| model_hash | Hash of AI model file | Yes |
| parameters_hash | Hash of screening parameters | Yes |
| molecule_smiles | SMILES representation | Yes |
| screening_type | Type of screening performed | Yes |
| strict_mode | Verification enforcement mode | Yes |
| timestamp | Execution timestamp | Yes |
| software_version | PureProtX version | Yes |

---

## 6. Blockchain Anchoring

### 6.1 PureChain Configuration

| Parameter | Value |
|-----------|-------|
| Network Name | PureChain |
| Chain ID | 900520900520 |
| Consensus | Proof of Authority (PoA) |
| Gas Cost | 0 (zero-fee) |
| RPC Endpoint | https://purechainnode.com:8547 |
| Native Currency | PCC |

### 6.2 Anchoring Protocol

**Algorithm 3: Blockchain Anchoring**
```
function AnchorToBlockchain(unified_record):
    // Generate unique job ID
    job_id = "unified_" + unified_record.record_id

    // Prepare transaction data
    tx_data = {
        "job_id": job_id,
        "molecule_id": unified_record.drug_screening.molecule_id,
        "smiles": unified_record.drug_screening.smiles,
        "result_hash": unified_record.master_hash,
        "additional_data": {
            "record_type": "unified_audit",
            "biomaterial_id": unified_record.biomaterial_provenance.material_id,
            "biomaterial_verified": unified_record.biomaterial_provenance.verification_status == "PASS",
            "credential_hash": unified_record.biomaterial_provenance.credential_hash
        }
    }

    // Submit transaction (zero gas fee)
    tx_hash = blockchain.submitTransaction(tx_data)

    // Wait for confirmation
    receipt = blockchain.waitForConfirmation(tx_hash)

    return (tx_hash, job_id)
```

### 6.3 Verification Protocol

Any party can independently verify a screening result:

```
function VerifyScreeningResult(job_id, claimed_record):
    // Retrieve on-chain data
    blockchain_data = blockchain.getScreeningResult(job_id)

    if blockchain_data is NULL:
        return FAIL("Job ID not found")

    // Recompute master hash from claimed record
    computed_hash = ComputeMasterHash(claimed_record)

    // Compare hashes
    if computed_hash != blockchain_data.result_hash:
        return FAIL("Hash mismatch")

    return PASS(blockchain_data.timestamp, blockchain_data.tx_hash)
```

---

## 7. Context-Aware Workflow

### 7.1 Complete Screening Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT                                         │
│  Molecule (SMILES) + Biomaterial ID + Model + Parameters        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: BIOMATERIAL VERIFICATION                               │
│  • Query BioPassport for credentials                            │
│  • Validate IDENTITY, QC_MYCO credentials                       │
│  • Check material status (not quarantined/revoked)              │
│  • Compute credential_hash                                       │
│  • Decision: PASS → continue | FAIL → abort (if strict mode)   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: AI SCREENING                                           │
│  • Generate molecular features (descriptors + fingerprints)     │
│  • Run SVR prediction                                           │
│  • Run Random Forest prediction                                 │
│  • Run Gradient Boosting prediction                             │
│  • Compute consensus (average)                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: UNIFIED AUDIT RECORD                                   │
│  • Link biomaterial credential_hash to screening results        │
│  • Include model_hash and parameters_hash                       │
│  • Compute master_hash (SHA-256 of canonical JSON)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: BLOCKCHAIN ANCHORING                                   │
│  • Submit master_hash to PureChain                              │
│  • Zero gas fee transaction                                     │
│  • Receive transaction hash                                     │
│  • Store audit record locally                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                        │
│  Verified Screening Result with Provenance Trail                │
│  • pIC50 prediction (consensus + individual)                    │
│  • Biomaterial verification status                              │
│  • Master hash                                                  │
│  • Blockchain transaction hash                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Strict vs Non-Strict Mode

| Mode | Behavior on Verification Failure |
|------|----------------------------------|
| Strict | Abort screening, return error |
| Non-Strict | Warn user, proceed with screening, mark as unverified |

---

## 8. Experimental Design

### 8.1 Experiment Categories

| Experiment | Objective | Key Metric |
|------------|-----------|------------|
| Context-Awareness | Prove different contexts → different hashes | Pass/Fail rate |
| Reproducibility | Prove identical inputs → identical hashes | Hash match rate (target: 100%) |
| Blockchain Performance | Measure system latency and throughput | p50/p95/p99 latency, tx/s |
| Provenance Completeness | Document all captured artifacts | Artifact count |
| Case Study | Demonstrate real pipeline execution | Logged/Reproducible/Verifiable |

### 8.2 Reproducibility Protocol

**Definition**: A system is *deterministically reproducible* if:

```
∀ input I, context C: Hash(Execute(I, C)) = Hash(Execute(I, C))
```

**Test Protocol**:
1. Execute workflow N times with identical inputs
2. Collect master_hash from each execution
3. Compute uniqueness: unique_hashes = |{h₁, h₂, ..., hₙ}|
4. Pass criterion: unique_hashes = 1

### 8.3 Performance Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Verification Latency | Time to verify biomaterial credentials | p50 < 50ms |
| Anchoring Latency | Time to anchor record to blockchain | p50 < 100ms |
| Full Workflow Latency | End-to-end time including AI screening | p50 < 500ms |
| Throughput | Transactions per second | > 10 tx/s |
| Gas Cost | Transaction fee | 0 (zero-fee) |

---

## 9. Implementation Details

### 9.1 Technology Stack

| Layer | Technology |
|-------|------------|
| Programming Language | Python 3.8+ |
| Blockchain Interface | Web3.py |
| Machine Learning | scikit-learn |
| Molecular Representation | RDKit |
| Hashing | hashlib (SHA-256) |
| Serialization | JSON (canonical) |
| CLI Framework | argparse |

### 9.2 Key Modules

```
PureProtX/
├── pureprot/                 # Core modules
│   ├── ai_model.py          # Consensus AI ensemble
│   ├── blockchain.py        # Blockchain auditor
│   ├── docking.py           # Molecular docking
│   └── data.py              # Data management
├── biopassport/             # BioPassport integration
│   ├── client.py            # Verification client
│   └── schemas.py           # Data structures
├── blockchain/              # Blockchain connectors
│   └── purechain_connector.py
├── experiments/             # Paper experiments
│   ├── context_awareness.py
│   ├── reproducibility.py
│   ├── blockchain_performance.py
│   ├── provenance_completeness.py
│   ├── case_study.py
│   └── visualizations.py
└── PureProt.py              # Main CLI
```

---

## 10. Summary

This methodology establishes a **context-aware drug discovery pipeline** that:

1. **Verifies biomaterial provenance** before screening via BioPassport integration
2. **Produces deterministic results** through canonical JSON serialization and SHA-256 hashing
3. **Anchors all provenance** to a zero-fee blockchain for immutable audit trails
4. **Links biomaterial credentials** to screening results via unified audit records
5. **Enables independent verification** of any screening result

The system addresses the critical need for **reproducibility** and **regulatory compliance** in computational drug discovery while maintaining practical throughput for real-world applications.

---

## References

- PureChain Network: https://purechainnode.com
- RDKit: https://www.rdkit.org
- scikit-learn: https://scikit-learn.org
- Web3.py: https://web3py.readthedocs.io
