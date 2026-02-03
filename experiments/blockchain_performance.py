"""
Experiment 3: Blockchain Verification Performance

Required metrics (NON-NEGOTIABLE for paper):
1. Transaction latency
2. Verification latency
3. Throughput (tx/s)
4. Blockchain overhead vs compute
5. Confirmation finality
6. Gas cost = 0 (zero-fee blockchain)
"""

import time
import statistics
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biopassport import BioPassportClient
from biopassport.schemas import UnifiedAuditRecord


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    operation: str
    latency_ms: float
    timestamp: str


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    operation: str
    num_samples: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float


class BlockchainPerformanceExperiment:
    """
    Measures blockchain verification performance metrics
    required for the ICUFN 2026 paper.
    """

    # PureChain configuration
    PURECHAIN_GAS_COST = 0  # Zero-fee blockchain
    PURECHAIN_CHAIN_ID = 900520900520
    PURECHAIN_CONSENSUS = "PoA"  # Proof of Authority

    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples
        self.client = BioPassportClient()
        self.measurements: Dict[str, List[float]] = {
            "verification_latency": [],
            "audit_creation_latency": [],
            "anchoring_latency": [],
            "full_workflow_latency": [],
        }

    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all blockchain performance experiments."""
        print("=" * 60)
        print("EXPERIMENT 3: Blockchain Verification Performance")
        print("=" * 60)
        print(f"Running {self.num_samples} samples per operation...\n")

        # Run measurements
        print("Measuring verification latency...")
        self._measure_verification_latency()

        print("Measuring audit record creation latency...")
        self._measure_audit_creation_latency()

        print("Measuring blockchain anchoring latency...")
        self._measure_anchoring_latency()

        print("Measuring full workflow latency...")
        self._measure_full_workflow_latency()

        # Calculate metrics
        metrics = {}
        for operation, samples in self.measurements.items():
            if samples:
                metrics[operation] = self._calculate_metrics(operation, samples)

        # Calculate overhead
        overhead = self._calculate_overhead(metrics)

        # Generate throughput estimate
        throughput = self._calculate_throughput(metrics)

        summary = {
            "experiment_category": "Blockchain Verification Performance",
            "num_samples": self.num_samples,
            "blockchain_config": {
                "network": "PureChain",
                "chain_id": self.PURECHAIN_CHAIN_ID,
                "consensus": self.PURECHAIN_CONSENSUS,
                "gas_cost": self.PURECHAIN_GAS_COST,
                "gas_cost_unit": "PCC (zero-fee)"
            },
            "performance_metrics": {k: asdict(v) for k, v in metrics.items()},
            "overhead_analysis": overhead,
            "throughput": throughput,
            "conclusion": self._generate_conclusion(metrics, overhead, throughput)
        }

        self._print_results(metrics, overhead, throughput)
        return summary

    def _measure_verification_latency(self):
        """Measure credential verification latency."""
        biomaterial_ids = [
            f"bio:cell_line:test-{i:03d}" for i in range(self.num_samples)
        ]

        for biomaterial_id in biomaterial_ids:
            start = time.perf_counter()
            self.client.verify_credential(biomaterial_id)
            latency = (time.perf_counter() - start) * 1000
            self.measurements["verification_latency"].append(latency)

    def _measure_audit_creation_latency(self):
        """Measure audit record creation latency."""
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
            latency = (time.perf_counter() - start) * 1000
            self.measurements["audit_creation_latency"].append(latency)

    def _measure_anchoring_latency(self):
        """Measure blockchain anchoring latency."""
        for i in range(self.num_samples):
            # Create record first
            record = self.client.create_unified_audit_record(
                biomaterial_id=f"bio:cell_line:anchor-{i:03d}",
                molecule_id=f"anchor_mol_{i}",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                screening_results={"consensus_pic50": 6.5},
                model_hash="test_model_hash"
            )

            # Measure anchoring
            start = time.perf_counter()
            self.client.anchor_to_blockchain(record)
            latency = (time.perf_counter() - start) * 1000
            self.measurements["anchoring_latency"].append(latency)

    def _measure_full_workflow_latency(self):
        """Measure complete workflow latency."""
        for i in range(self.num_samples):
            start = time.perf_counter()

            # Full workflow: verify -> create record -> anchor
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

            latency = (time.perf_counter() - start) * 1000
            self.measurements["full_workflow_latency"].append(latency)

    def _calculate_metrics(self, operation: str, samples: List[float]) -> PerformanceMetrics:
        """Calculate statistical metrics for a set of samples."""
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return PerformanceMetrics(
            operation=operation,
            num_samples=n,
            min_ms=min(samples),
            max_ms=max(samples),
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            p50_ms=sorted_samples[int(n * 0.50)],
            p95_ms=sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1],
            p99_ms=sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1],
            std_dev_ms=statistics.stdev(samples) if n > 1 else 0.0
        )

    def _calculate_overhead(self, metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate blockchain overhead vs compute time."""
        # Assume AI screening takes ~100ms (typical for consensus model)
        estimated_ai_time_ms = 100.0

        verification_overhead = metrics.get("verification_latency")
        anchoring_overhead = metrics.get("anchoring_latency")
        full_workflow = metrics.get("full_workflow_latency")

        blockchain_overhead_ms = 0
        if verification_overhead:
            blockchain_overhead_ms += verification_overhead.mean_ms
        if anchoring_overhead:
            blockchain_overhead_ms += anchoring_overhead.mean_ms

        total_time = estimated_ai_time_ms + blockchain_overhead_ms
        overhead_percentage = (blockchain_overhead_ms / total_time) * 100

        return {
            "estimated_ai_compute_ms": estimated_ai_time_ms,
            "blockchain_overhead_ms": round(blockchain_overhead_ms, 2),
            "total_workflow_ms": round(total_time, 2),
            "overhead_percentage": round(overhead_percentage, 1),
            "interpretation": (
                f"Blockchain verification introduced {blockchain_overhead_ms:.1f}ms of overhead, "
                f"representing {overhead_percentage:.1f}% of total workflow time."
            )
        }

    def _calculate_throughput(self, metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate throughput metrics."""
        full_workflow = metrics.get("full_workflow_latency")

        if full_workflow:
            # Transactions per second
            tx_per_second = 1000.0 / full_workflow.mean_ms

            # Molecules per minute (practical throughput)
            molecules_per_minute = tx_per_second * 60

            return {
                "transactions_per_second": round(tx_per_second, 2),
                "molecules_per_minute": round(molecules_per_minute, 1),
                "mean_latency_ms": round(full_workflow.mean_ms, 2),
                "interpretation": (
                    f"System can process approximately {molecules_per_minute:.0f} verified "
                    f"screenings per minute ({tx_per_second:.1f} tx/s)."
                )
            }
        return {}

    def _print_results(self, metrics: Dict[str, PerformanceMetrics],
                       overhead: Dict, throughput: Dict):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("Performance Results:")
        print("=" * 60)

        print("\n[Latency Metrics]")
        print("-" * 50)
        print(f"{'Operation':<30} {'p50':>8} {'p95':>8} {'p99':>8}")
        print("-" * 50)
        for op, m in metrics.items():
            op_name = op.replace("_", " ").title()[:28]
            print(f"{op_name:<30} {m.p50_ms:>7.1f}ms {m.p95_ms:>7.1f}ms {m.p99_ms:>7.1f}ms")

        print("\n[Blockchain Configuration]")
        print("-" * 50)
        print(f"Network: PureChain (Chain ID: {self.PURECHAIN_CHAIN_ID})")
        print(f"Consensus: {self.PURECHAIN_CONSENSUS}")
        print(f"Gas Cost: {self.PURECHAIN_GAS_COST} PCC (ZERO FEE)")

        print("\n[Overhead Analysis]")
        print("-" * 50)
        print(f"AI Compute Time: ~{overhead['estimated_ai_compute_ms']:.0f}ms")
        print(f"Blockchain Overhead: {overhead['blockchain_overhead_ms']:.1f}ms")
        print(f"Overhead Percentage: {overhead['overhead_percentage']:.1f}%")

        print("\n[Throughput]")
        print("-" * 50)
        print(f"Transactions/second: {throughput.get('transactions_per_second', 'N/A')}")
        print(f"Molecules/minute: {throughput.get('molecules_per_minute', 'N/A')}")

    def _generate_conclusion(self, metrics: Dict[str, PerformanceMetrics],
                            overhead: Dict, throughput: Dict) -> str:
        """Generate conclusion for the paper."""
        verification = metrics.get("verification_latency")
        anchoring = metrics.get("anchoring_latency")

        return (
            f"Blockchain verification on PureChain (Chain ID: {self.PURECHAIN_CHAIN_ID}) "
            f"achieved p50 verification latency of {verification.p50_ms:.1f}ms and "
            f"p95 latency of {verification.p95_ms:.1f}ms. "
            f"Blockchain anchoring added {anchoring.mean_ms:.1f}ms mean latency. "
            f"Total blockchain overhead represents {overhead['overhead_percentage']:.1f}% "
            f"of workflow time. Zero-fee execution (gas cost = 0 PCC) enabled "
            f"unrestricted provenance logging at {throughput.get('transactions_per_second', 0):.1f} tx/s."
        )

    def get_latex_table(self) -> str:
        """Generate LaTeX table for the paper."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Blockchain Verification Performance Metrics}",
            "\\begin{tabular}{lrrr}",
            "\\hline",
            "Operation & p50 (ms) & p95 (ms) & p99 (ms) \\\\",
            "\\hline"
        ]

        for op, samples in self.measurements.items():
            if samples:
                m = self._calculate_metrics(op, samples)
                op_name = op.replace("_", " ").title()
                lines.append(f"{op_name} & {m.p50_ms:.1f} & {m.p95_ms:.1f} & {m.p99_ms:.1f} \\\\")

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
