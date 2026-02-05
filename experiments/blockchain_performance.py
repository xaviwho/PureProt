"""
Experiment 3: Blockchain Verification Performance

Required metrics (NON-NEGOTIABLE for paper):
1. Transaction latency (submit vs finality separated)
2. Verification latency (local operations in µs)
3. Throughput (tx/s) with clear 1:1 molecule-to-tx mapping
4. Blockchain overhead vs compute
5. Confirmation finality
6. Gas cost = 0 (zero-fee blockchain)
7. Transaction receipt example
"""

import time
import statistics
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopassport import BioPassportClient
from biopassport.schemas import UnifiedAuditRecord


@dataclass
class LatencyMeasurement:
    """Single latency measurement with submit/finality breakdown."""
    operation: str
    total_latency_ms: float
    submit_latency_ms: Optional[float] = None
    finality_latency_ms: Optional[float] = None
    timestamp: str = ""


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics with µs precision for fast operations."""
    operation: str
    num_samples: int
    min_us: float  # Microseconds for precision
    max_us: float
    mean_us: float
    median_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    std_dev_us: float
    # Also provide ms for slower operations
    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass
class TransactionReceipt:
    """Captured transaction receipt for paper evidence."""
    tx_hash: str
    block_number: int
    status: str
    gas_used: int
    submit_latency_ms: float
    finality_latency_ms: float
    total_latency_ms: float
    timestamp: str


class BlockchainPerformanceExperiment:
    """
    Measures blockchain verification performance metrics
    required for the ICUFN 2026 paper.

    Workflow Definitions:
    ---------------------
    (A) AUDIT PATH: verify biomaterial -> create record -> anchor hash -> return receipt
        - This is what we benchmark
        - Does NOT include AI inference or docking

    (B) SCREENING PATH: AUDIT PATH + AI inference (+ optional docking)
        - AI inference is treated as a separate workload
        - Docking is an optional module, not included in benchmarks

    Throughput Note:
    ----------------
    1 molecule = 1 blockchain transaction (1:1 mapping)
    No batching is used in this configuration.
    """

    # PureChain configuration
    PURECHAIN_GAS_COST = 0  # Zero-fee blockchain
    PURECHAIN_CHAIN_ID = 900520900520
    PURECHAIN_CONSENSUS = "PoA"  # Proof of Authority

    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
        self.client = BioPassportClient()

        # Measurements in microseconds for precision
        self.measurements: Dict[str, List[float]] = {
            "verification_latency_us": [],      # Local policy + hash (µs)
            "audit_creation_latency_us": [],    # Serialization + SHA-256 (µs)
            "tx_submit_latency_ms": [],         # RPC call until tx hash returned
            "tx_finality_latency_ms": [],       # Wait for block confirmation
            "anchoring_total_latency_ms": [],   # Submit + finality combined
            "full_workflow_latency_ms": [],     # Complete audit path
        }

        # Capture sample transaction receipts
        self.sample_receipts: List[TransactionReceipt] = []

    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all blockchain performance experiments."""
        print("=" * 60)
        print("EXPERIMENT 3: Blockchain Verification Performance")
        print("=" * 60)
        print(f"Running {self.num_samples} samples per operation...")
        print("\nWorkflow Definition: AUDIT PATH")
        print("  verify -> create record -> anchor -> receipt")
        print("  (AI inference/docking NOT included)\n")

        # Run measurements
        print("Measuring verification latency (local operations)...")
        self._measure_verification_latency()

        print("Measuring audit record creation latency (local)...")
        self._measure_audit_creation_latency()

        print("Measuring blockchain anchoring (submit + finality)...")
        self._measure_anchoring_latency_detailed()

        print("Measuring full audit path workflow...")
        self._measure_full_workflow_latency()

        # Calculate metrics
        metrics = self._calculate_all_metrics()

        # Calculate overhead
        overhead = self._calculate_overhead(metrics)

        # Generate throughput estimate
        throughput = self._calculate_throughput(metrics)

        # Get sample receipt
        sample_receipt = self.sample_receipts[0] if self.sample_receipts else None

        summary = {
            "experiment_category": "Blockchain Verification Performance",
            "num_samples": self.num_samples,
            "workflow_definition": {
                "name": "AUDIT PATH",
                "description": "verify biomaterial -> create record -> anchor hash -> return receipt",
                "includes": ["credential verification (local)", "audit record creation (local)", "blockchain anchoring"],
                "excludes": ["AI inference", "molecular docking"],
                "molecule_to_tx_ratio": "1:1 (no batching)"
            },
            "blockchain_config": {
                "network": "PureChain",
                "chain_id": self.PURECHAIN_CHAIN_ID,
                "consensus": self.PURECHAIN_CONSENSUS,
                "gas_cost": self.PURECHAIN_GAS_COST,
                "gas_cost_unit": "PCC (zero-fee)"
            },
            "performance_metrics": metrics,
            "latency_breakdown": {
                "local_operations": {
                    "verification": "Local policy evaluation + SHA-256 hash check (no chain read)",
                    "audit_creation": "Canonical JSON serialization + SHA-256 hashing"
                },
                "blockchain_operations": {
                    "tx_submit": "Sign transaction + RPC send_raw_transaction call",
                    "tx_finality": "Wait for block inclusion + receipt confirmation"
                }
            },
            "overhead_analysis": overhead,
            "throughput": throughput,
            "sample_transaction_receipt": asdict(sample_receipt) if sample_receipt else None,
            "conclusion": self._generate_conclusion(metrics, overhead, throughput)
        }

        self._print_results(metrics, overhead, throughput, sample_receipt)
        return summary

    def _measure_verification_latency(self):
        """
        Measure credential verification latency.

        This is a LOCAL operation:
        - Policy evaluation against cached/deterministic rules
        - SHA-256 hash computation
        - NO blockchain read involved
        """
        biomaterial_ids = [
            f"bio:cell_line:test-{i:03d}" for i in range(self.num_samples)
        ]

        for biomaterial_id in biomaterial_ids:
            start = time.perf_counter()
            self.client.verify_credential(biomaterial_id)
            latency_us = (time.perf_counter() - start) * 1_000_000  # Microseconds
            self.measurements["verification_latency_us"].append(latency_us)

    def _measure_audit_creation_latency(self):
        """
        Measure audit record creation latency.

        This is a LOCAL operation:
        - Canonical JSON serialization
        - SHA-256 hash computation
        - NO network/disk IO
        """
        for i in range(self.num_samples):
            start = time.perf_counter()
            self.client.create_unified_audit_record(
                biomaterial_id=f"bio:cell_line:test-{i:03d}",
                molecule_id=f"mol_{i}",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                screening_results={"consensus_pic50": 6.5 + (i * 0.01)},
                model_hash="test_model_hash",
                parameters={"test": True}
            )
            latency_us = (time.perf_counter() - start) * 1_000_000  # Microseconds
            self.measurements["audit_creation_latency_us"].append(latency_us)

    def _measure_anchoring_latency_detailed(self):
        """
        Measure blockchain anchoring with submit/finality breakdown.

        Submit latency: Time to sign and send transaction (RPC returns tx hash)
        Finality latency: Time to wait for block confirmation (receipt)
        """
        for i in range(self.num_samples):
            # Create record first
            record = self.client.create_unified_audit_record(
                biomaterial_id=f"bio:cell_line:anchor-{i:03d}",
                molecule_id=f"anchor_mol_{i}",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                screening_results={"consensus_pic50": 6.5},
                model_hash="test_model_hash"
            )

            # Measure anchoring with detailed timing
            submit_ms, finality_ms, total_ms, receipt_data = self._anchor_with_timing(record)

            self.measurements["tx_submit_latency_ms"].append(submit_ms)
            self.measurements["tx_finality_latency_ms"].append(finality_ms)
            self.measurements["anchoring_total_latency_ms"].append(total_ms)

            # Capture first few receipts as examples
            if len(self.sample_receipts) < 3 and receipt_data:
                self.sample_receipts.append(receipt_data)

    def _anchor_with_timing(self, record: UnifiedAuditRecord) -> tuple:
        """Anchor with separate submit/finality timing."""
        try:
            w3 = self.client._w3
            contract = self.client._contract
            account = self.client._account
            private_key = self.client._private_key

            # Prepare transaction
            result_hash = bytes.fromhex(record.master_hash[:64])
            molecule_hash = bytes.fromhex(
                hashlib.sha256(record.smiles.encode()).hexdigest()
            )

            nonce = w3.eth.get_transaction_count(account.address, 'pending')

            tx = contract.functions.recordScreeningResult(
                result_hash,
                molecule_hash,
                record.molecule_id
            ).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': 300000,
                'gasPrice': 0,
                'chainId': self.client.chain_id
            })

            # SUBMIT PHASE: Sign and send
            submit_start = time.perf_counter()
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            submit_end = time.perf_counter()
            submit_ms = (submit_end - submit_start) * 1000

            # FINALITY PHASE: Wait for confirmation
            finality_start = time.perf_counter()
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            finality_end = time.perf_counter()
            finality_ms = (finality_end - finality_start) * 1000

            total_ms = submit_ms + finality_ms

            # Create receipt data
            receipt_data = TransactionReceipt(
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                status="SUCCESS" if receipt.status == 1 else "FAILED",
                gas_used=receipt.gasUsed,
                submit_latency_ms=round(submit_ms, 2),
                finality_latency_ms=round(finality_ms, 2),
                total_latency_ms=round(total_ms, 2),
                timestamp=datetime.now().isoformat()
            )

            record.purechain_tx = tx_hash.hex()

            return submit_ms, finality_ms, total_ms, receipt_data

        except Exception as e:
            print(f"  Warning: Anchoring error: {e}")
            # Retry with fresh nonce
            time.sleep(0.5)
            return self._anchor_with_timing(record)

    def _measure_full_workflow_latency(self):
        """
        Measure complete AUDIT PATH workflow latency.

        AUDIT PATH = verify -> create record -> anchor -> receipt

        Note: This does NOT include AI inference or docking.
        Those are separate workloads not part of the audit path.
        """
        for i in range(self.num_samples):
            start = time.perf_counter()

            # Full audit path workflow
            can_proceed, _ = self.client.verified_screen_check(
                f"bio:cell_line:full-{i:03d}"
            )

            record = self.client.create_unified_audit_record(
                biomaterial_id=f"bio:cell_line:full-{i:03d}",
                molecule_id=f"full_mol_{i}",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                screening_results={"consensus_pic50": 6.5, "verified": can_proceed},
                model_hash="test_model_hash"
            )

            self.client.anchor_to_blockchain(record)

            latency_ms = (time.perf_counter() - start) * 1000
            self.measurements["full_workflow_latency_ms"].append(latency_ms)

    def _calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate metrics with appropriate units."""
        metrics = {}

        # Local operations in microseconds
        for op in ["verification_latency_us", "audit_creation_latency_us"]:
            samples = self.measurements.get(op, [])
            if samples:
                metrics[op] = self._calculate_metrics_us(op, samples)

        # Blockchain operations in milliseconds
        for op in ["tx_submit_latency_ms", "tx_finality_latency_ms",
                   "anchoring_total_latency_ms", "full_workflow_latency_ms"]:
            samples = self.measurements.get(op, [])
            if samples:
                metrics[op] = self._calculate_metrics_ms(op, samples)

        return metrics

    def _calculate_metrics_us(self, operation: str, samples: List[float]) -> Dict[str, Any]:
        """Calculate metrics in microseconds for local operations."""
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return {
            "operation": operation,
            "unit": "microseconds (µs)",
            "num_samples": n,
            "min_us": round(min(samples), 1),
            "max_us": round(max(samples), 1),
            "mean_us": round(statistics.mean(samples), 1),
            "median_us": round(statistics.median(samples), 1),
            "p50_us": round(sorted_samples[int(n * 0.50)], 1),
            "p95_us": round(sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1], 1),
            "p99_us": round(sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1], 1),
            "std_dev_us": round(statistics.stdev(samples) if n > 1 else 0.0, 1),
            # Also provide ms for comparison
            "mean_ms": round(statistics.mean(samples) / 1000, 3),
            "p50_ms": round(sorted_samples[int(n * 0.50)] / 1000, 3),
        }

    def _calculate_metrics_ms(self, operation: str, samples: List[float]) -> Dict[str, Any]:
        """Calculate metrics in milliseconds for blockchain operations."""
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return {
            "operation": operation,
            "unit": "milliseconds (ms)",
            "num_samples": n,
            "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1),
            "mean_ms": round(statistics.mean(samples), 1),
            "median_ms": round(statistics.median(samples), 1),
            "p50_ms": round(sorted_samples[int(n * 0.50)], 1),
            "p95_ms": round(sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1], 1),
            "p99_ms": round(sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1], 1),
            "std_dev_ms": round(statistics.stdev(samples) if n > 1 else 0.0, 1),
        }

    def _calculate_overhead(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate blockchain overhead vs local compute time."""
        # Local operations (in ms)
        verification = metrics.get("verification_latency_us", {})
        audit_creation = metrics.get("audit_creation_latency_us", {})

        local_compute_ms = (
            verification.get("mean_ms", 0) +
            audit_creation.get("mean_ms", 0)
        )

        # Blockchain operations
        anchoring = metrics.get("anchoring_total_latency_ms", {})
        blockchain_ms = anchoring.get("mean_ms", 0)

        total_audit_path_ms = local_compute_ms + blockchain_ms

        if total_audit_path_ms > 0:
            blockchain_overhead_pct = (blockchain_ms / total_audit_path_ms) * 100
            local_compute_pct = (local_compute_ms / total_audit_path_ms) * 100
        else:
            blockchain_overhead_pct = 0
            local_compute_pct = 0

        return {
            "workflow": "AUDIT PATH (excludes AI inference/docking)",
            "local_compute_ms": round(local_compute_ms, 3),
            "local_compute_pct": round(local_compute_pct, 2),
            "blockchain_overhead_ms": round(blockchain_ms, 1),
            "blockchain_overhead_pct": round(blockchain_overhead_pct, 1),
            "total_audit_path_ms": round(total_audit_path_ms, 1),
            "note": "AI inference (~100ms typical) and docking are separate workloads, not included here"
        }

    def _calculate_throughput(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate throughput with clear molecule-to-tx mapping."""
        full_workflow = metrics.get("full_workflow_latency_ms", {})

        if full_workflow:
            mean_latency = full_workflow.get("mean_ms", 0)
            if mean_latency > 0:
                tx_per_second = 1000.0 / mean_latency
                molecules_per_minute = tx_per_second * 60

                return {
                    "transactions_per_second": round(tx_per_second, 2),
                    "molecules_per_minute": round(molecules_per_minute, 1),
                    "molecule_to_tx_ratio": "1:1 (one blockchain transaction per molecule)",
                    "batching": "None (sequential processing)",
                    "mean_latency_ms": round(mean_latency, 1),
                    "interpretation": (
                        f"System processes {tx_per_second:.2f} verified screenings/sec "
                        f"(~{molecules_per_minute:.0f}/min). Each molecule results in "
                        f"exactly one blockchain transaction (no batching)."
                    )
                }
        return {}

    def _print_results(self, metrics: Dict[str, Any], overhead: Dict,
                       throughput: Dict, sample_receipt: Optional[TransactionReceipt]):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("Performance Results:")
        print("=" * 70)

        print("\n[Local Operations (µs) - No Chain Read]")
        print("-" * 60)
        print(f"{'Operation':<35} {'p50':>10} {'p95':>10} {'p99':>10}")
        print("-" * 60)
        for op in ["verification_latency_us", "audit_creation_latency_us"]:
            m = metrics.get(op, {})
            if m:
                op_name = op.replace("_us", "").replace("_", " ").title()[:33]
                print(f"{op_name:<35} {m['p50_us']:>9.1f}µs {m['p95_us']:>9.1f}µs {m['p99_us']:>9.1f}µs")

        print("\n[Blockchain Operations (ms) - Submit vs Finality]")
        print("-" * 60)
        print(f"{'Operation':<35} {'p50':>10} {'p95':>10} {'p99':>10}")
        print("-" * 60)
        for op in ["tx_submit_latency_ms", "tx_finality_latency_ms", "anchoring_total_latency_ms"]:
            m = metrics.get(op, {})
            if m:
                op_name = op.replace("_ms", "").replace("_", " ").title()[:33]
                print(f"{op_name:<35} {m['p50_ms']:>9.1f}ms {m['p95_ms']:>9.1f}ms {m['p99_ms']:>9.1f}ms")

        print("\n[Blockchain Configuration]")
        print("-" * 60)
        print(f"Network: PureChain (Chain ID: {self.PURECHAIN_CHAIN_ID})")
        print(f"Consensus: {self.PURECHAIN_CONSENSUS}")
        print(f"Gas Cost: {self.PURECHAIN_GAS_COST} PCC (ZERO FEE)")

        print("\n[Overhead Analysis - AUDIT PATH]")
        print("-" * 60)
        print(f"Local Compute: {overhead['local_compute_ms']:.3f}ms ({overhead['local_compute_pct']:.1f}%)")
        print(f"Blockchain Overhead: {overhead['blockchain_overhead_ms']:.1f}ms ({overhead['blockchain_overhead_pct']:.1f}%)")
        print(f"Note: {overhead['note']}")

        print("\n[Throughput - 1 molecule = 1 transaction]")
        print("-" * 60)
        print(f"Transactions/second: {throughput.get('transactions_per_second', 'N/A')}")
        print(f"Molecules/minute: {throughput.get('molecules_per_minute', 'N/A')}")
        print(f"Batching: {throughput.get('batching', 'None')}")

        if sample_receipt:
            print("\n[Sample Transaction Receipt]")
            print("-" * 60)
            print(f"TX Hash: {sample_receipt.tx_hash}")
            print(f"Block Number: {sample_receipt.block_number}")
            print(f"Status: {sample_receipt.status}")
            print(f"Gas Used: {sample_receipt.gas_used}")
            print(f"Submit Latency: {sample_receipt.submit_latency_ms}ms")
            print(f"Finality Latency: {sample_receipt.finality_latency_ms}ms")
            print(f"Total Latency: {sample_receipt.total_latency_ms}ms")

    def _generate_conclusion(self, metrics: Dict[str, Any],
                            overhead: Dict, throughput: Dict) -> str:
        """Generate conclusion for the paper."""
        verification = metrics.get("verification_latency_us", {})
        audit_creation = metrics.get("audit_creation_latency_us", {})
        submit = metrics.get("tx_submit_latency_ms", {})
        finality = metrics.get("tx_finality_latency_ms", {})
        total = metrics.get("anchoring_total_latency_ms", {})

        return (
            f"On PureChain (Chain ID: {self.PURECHAIN_CHAIN_ID}), local operations "
            f"(verification + audit creation) completed in {verification.get('p50_us', 0):.0f}µs + "
            f"{audit_creation.get('p50_us', 0):.0f}µs (p50). "
            f"Blockchain anchoring: tx submit p50={submit.get('p50_ms', 0):.0f}ms, "
            f"block finality p50={finality.get('p50_ms', 0):.0f}ms, "
            f"total p50={total.get('p50_ms', 0):.0f}ms. "
            f"Blockchain represents {overhead['blockchain_overhead_pct']:.1f}% of audit path time. "
            f"Throughput: {throughput.get('transactions_per_second', 0):.2f} tx/s "
            f"(1 molecule = 1 tx, no batching). Zero-fee execution confirmed."
        )

    def get_latex_table(self) -> str:
        """Generate LaTeX table for the paper with proper units."""
        metrics = self._calculate_all_metrics()

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Blockchain Verification Performance Metrics. Local operations (verification, audit creation) are measured in microseconds; blockchain operations are measured in milliseconds.}",
            "\\begin{tabular}{llrrr}",
            "\\hline",
            "Operation & Unit & p50 & p95 & p99 \\\\",
            "\\hline"
        ]

        # Local operations in µs
        for op in ["verification_latency_us", "audit_creation_latency_us"]:
            m = metrics.get(op, {})
            if m:
                op_name = op.replace("_us", "").replace("_", " ").title()
                lines.append(f"{op_name} & µs & {m['p50_us']:.1f} & {m['p95_us']:.1f} & {m['p99_us']:.1f} \\\\")

        # Blockchain operations in ms
        for op in ["tx_submit_latency_ms", "tx_finality_latency_ms", "anchoring_total_latency_ms"]:
            m = metrics.get(op, {})
            if m:
                op_name = op.replace("_ms", "").replace("_", " ").title()
                lines.append(f"{op_name} & ms & {m['p50_ms']:.1f} & {m['p95_ms']:.1f} & {m['p99_ms']:.1f} \\\\")

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:performance}",
            "\\end{table}"
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    experiment = BlockchainPerformanceExperiment(num_samples=50)
    results = experiment.run_all_experiments()

    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)
    print(experiment.get_latex_table())
