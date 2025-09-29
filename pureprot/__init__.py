"""
PureProtX: Modular CLI Protocol for Blockchain-Audited Consensus AI and Docking-Based Virtual Screening

This package provides modular components for AI-blockchain enabled drug discovery.
"""

__version__ = "1.0.0"
__author__ = "PureProtX Team"

from .ai_model import ConsensusAIModel
from .blockchain import BlockchainAuditor
from .docking import DockingEngine
from .data import DataManager

__all__ = ['ConsensusAIModel', 'BlockchainAuditor', 'DockingEngine', 'DataManager']
