"""
BioPassport Integration Module for PureProtX

This module provides integration with the BioPassport biomaterial provenance
verification system, enabling context-aware drug discovery with verified
biomaterial credentials.
"""

from .client import BioPassportClient
from .schemas import (
    CredentialType,
    VerificationStatus,
    BiomaterialCredential,
    VerificationResult,
    UnifiedAuditRecord
)

__all__ = [
    'BioPassportClient',
    'CredentialType',
    'VerificationStatus',
    'BiomaterialCredential',
    'VerificationResult',
    'UnifiedAuditRecord'
]
