"""
PureProtX Experiments Module for ICUFN 2026 Paper

This module contains all experiments required to generate results
for the paper: "Context-Aware Drug Discovery with Zero-Fee
Blockchain-Verified Biomaterial Provenance"

Experiment Categories:
1. Context-Awareness Validation
2. Deterministic Reproducibility
3. Blockchain Verification Performance
4. Provenance Completeness
5. Drug Discovery Case Study
"""

from .context_awareness import ContextAwarenessExperiment
from .reproducibility import ReproducibilityExperiment
from .blockchain_performance import BlockchainPerformanceExperiment
from .provenance_completeness import ProvenanceCompletenessExperiment
from .case_study import DrugDiscoveryCaseStudy
from .visualizations import PaperVisualizations, generate_visualizations

__all__ = [
    'ContextAwarenessExperiment',
    'ReproducibilityExperiment',
    'BlockchainPerformanceExperiment',
    'ProvenanceCompletenessExperiment',
    'DrugDiscoveryCaseStudy',
    'PaperVisualizations',
    'generate_visualizations'
]
