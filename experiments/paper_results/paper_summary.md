# Experimental Results Summary

**Paper**: Context-Aware Drug Discovery with Zero-Fee Blockchain-Verified Biomaterial Provenance
**Conference**: ICUFN 2026
**Generated**: 2026-02-04T20:00:17.284528

---

## 1. Context-Awareness Validation

**Result**: 6/6 tests passed (100.0%)

**Key Finding**: All 6 context-awareness tests passed. The system correctly produces identical hashes for identical contexts and distinct hashes when any contextual parameter changes. The system captures 9 distinct context fields in each provenance record.

---

## 2. Deterministic Reproducibility

**Result**: 100% hash reproducibility rate

**Key Finding**: Across 40 re-executions (10 runs x 4 test categories), PureProtX achieved a 100% hash reproducibility rate. Identical inputs under identical context consistently produce identical provenance hashes, demonstrating deterministic behavior suitable for regulatory compliance and scientific reproducibility.

---

## 3. Blockchain Verification Performance

**Configuration**:
- Network: PureChain (Chain ID: 900520900520)
- Consensus: PoA
- Gas Cost: **0 PCC (ZERO FEE)**

**Key Finding**: Blockchain verification on PureChain (Chain ID: 900520900520) achieved p50 verification latency of 0.0ms and p95 latency of 0.0ms. Blockchain anchoring added 1707.7ms mean latency. Total blockchain overhead represents 94.5% of workflow time. Zero-fee execution (gas cost = 0 PCC) enabled unrestricted provenance logging at 0.5 tx/s.

---

## 4. Provenance Completeness

**Artifacts Captured**: 7

**Key Finding**: The system captures 7 distinct artifacts in each provenance record, all anchored on-chain via SHA-256 hashing. Each record links biomaterial credentials (from BioPassport) to drug screening results (from PureProtX), creating a complete, independently verifiable audit trail. The master hash enables single-point verification of the entire provenance chain.

---

## 5. Drug Discovery Case Study

**Molecules Screened**: 5

**Key Finding**: The case study demonstrates successful execution of a context-aware drug screening pipeline on 5 molecules. All screening results were logged (100%), reproducible (100%), and independently verifiable (100%) via blockchain-anchored provenance records. Biomaterial verification succeeded for 100% of screenings, demonstrating the integration between BioPassport credential verification and PureProtX AI screening within a unified, auditable workflow.

---

## Summary for Reviewers

This system demonstrates:

1. **Context-Awareness**: Different contexts produce different provenance records; identical contexts produce identical hashes
2. **Deterministic Reproducibility**: 100% hash match rate across re-executions
3. **Blockchain Performance**: Zero-fee execution with measured latency metrics
4. **Provenance Completeness**: Full audit trail linking biomaterials to screening results
5. **Practical Application**: Successfully executed drug screening pipeline with verifiable results

