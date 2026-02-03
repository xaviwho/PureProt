"""
Data Schemas for BioPassport Integration

Defines credential types, verification results, and unified audit records
for the context-aware drug discovery pipeline.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json


class CredentialType(Enum):
    """Supported BioPassport credential types."""
    IDENTITY = "IDENTITY"           # Cell line STR profile or plasmid sequence fingerprint
    QC_MYCO = "QC_MYCO"             # Mycoplasma test result
    TRANSFER = "TRANSFER"           # Chain-of-custody event
    USAGE_RIGHTS = "USAGE_RIGHTS"   # MTA restrictions and expiration
    SCREENING_TARGET = "SCREENING_TARGET"  # Target protein validation for drug screening


class VerificationStatus(Enum):
    """Verification outcome status."""
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class BiomaterialCredential:
    """Represents a BioPassport biomaterial credential."""
    material_id: str                    # e.g., "bio:cell_line:<uuid>"
    credential_id: str                  # e.g., "cred:<uuid>"
    credential_type: CredentialType
    commitment_hash: str                # SHA-256 of canonical credential JSON
    issuer_id: str                      # Organization that issued the credential
    issued_at: datetime
    valid_until: Optional[datetime] = None
    artifact_refs: List[Dict[str, str]] = field(default_factory=list)
    signature_ref: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'material_id': self.material_id,
            'credential_id': self.credential_id,
            'credential_type': self.credential_type.value,
            'commitment_hash': self.commitment_hash,
            'issuer_id': self.issuer_id,
            'issued_at': self.issued_at.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'artifact_refs': self.artifact_refs,
            'signature_ref': self.signature_ref
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiomaterialCredential':
        """Create from dictionary."""
        return cls(
            material_id=data['material_id'],
            credential_id=data['credential_id'],
            credential_type=CredentialType(data['credential_type']),
            commitment_hash=data['commitment_hash'],
            issuer_id=data['issuer_id'],
            issued_at=datetime.fromisoformat(data['issued_at']),
            valid_until=datetime.fromisoformat(data['valid_until']) if data.get('valid_until') else None,
            artifact_refs=data.get('artifact_refs', []),
            signature_ref=data.get('signature_ref')
        )


@dataclass
class VerificationResult:
    """Result of a BioPassport credential verification."""
    material_id: str
    status: VerificationStatus
    verified_at: datetime
    credential_hash: Optional[str] = None
    policy_checks: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None
    on_chain_tx: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'material_id': self.material_id,
            'status': self.status.value,
            'verified_at': self.verified_at.isoformat(),
            'credential_hash': self.credential_hash,
            'policy_checks': self.policy_checks,
            'error_message': self.error_message,
            'on_chain_tx': self.on_chain_tx
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary."""
        return cls(
            material_id=data['material_id'],
            status=VerificationStatus(data['status']),
            verified_at=datetime.fromisoformat(data['verified_at']),
            credential_hash=data.get('credential_hash'),
            policy_checks=data.get('policy_checks', {}),
            error_message=data.get('error_message'),
            on_chain_tx=data.get('on_chain_tx')
        )


@dataclass
class UnifiedAuditRecord:
    """
    Unified audit record combining BioPassport credential verification
    with PureProtX drug screening results.

    This is the core data structure for context-aware verifiable drug discovery.
    """
    # Identifiers
    record_id: str
    timestamp: datetime

    # Biomaterial provenance (from BioPassport)
    biomaterial_id: str
    biomaterial_verification: VerificationResult
    biomaterial_credential_hash: str

    # Drug screening (from PureProtX)
    molecule_id: str
    smiles: str
    screening_results: Dict[str, Any]
    model_hash: Optional[str] = None
    parameters_hash: Optional[str] = None

    # Blockchain anchoring
    purechain_tx: Optional[str] = None
    master_hash: Optional[str] = None

    def __post_init__(self):
        """Calculate master hash after initialization."""
        if self.master_hash is None:
            self.master_hash = self.calculate_master_hash()

    def calculate_master_hash(self) -> str:
        """
        Calculate the master hash of the unified audit record.
        This hash anchors the complete provenance chain to the blockchain.
        """
        hash_components = {
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'biomaterial_id': self.biomaterial_id,
            'biomaterial_credential_hash': self.biomaterial_credential_hash,
            'biomaterial_verification_status': self.biomaterial_verification.status.value,
            'molecule_id': self.molecule_id,
            'smiles': self.smiles,
            'screening_results': self.screening_results,
            'model_hash': self.model_hash,
            'parameters_hash': self.parameters_hash
        }

        canonical_json = json.dumps(hash_components, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def is_valid_provenance(self) -> bool:
        """Check if the biomaterial provenance is valid for screening."""
        return self.biomaterial_verification.is_valid()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'biomaterial_id': self.biomaterial_id,
            'biomaterial_verification': self.biomaterial_verification.to_dict(),
            'biomaterial_credential_hash': self.biomaterial_credential_hash,
            'molecule_id': self.molecule_id,
            'smiles': self.smiles,
            'screening_results': self.screening_results,
            'model_hash': self.model_hash,
            'parameters_hash': self.parameters_hash,
            'purechain_tx': self.purechain_tx,
            'master_hash': self.master_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedAuditRecord':
        """Create from dictionary."""
        return cls(
            record_id=data['record_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            biomaterial_id=data['biomaterial_id'],
            biomaterial_verification=VerificationResult.from_dict(data['biomaterial_verification']),
            biomaterial_credential_hash=data['biomaterial_credential_hash'],
            molecule_id=data['molecule_id'],
            smiles=data['smiles'],
            screening_results=data['screening_results'],
            model_hash=data.get('model_hash'),
            parameters_hash=data.get('parameters_hash'),
            purechain_tx=data.get('purechain_tx'),
            master_hash=data.get('master_hash')
        )

    def get_audit_summary(self) -> str:
        """Generate human-readable audit summary."""
        return f"""
=== Unified Audit Record ===
Record ID: {self.record_id}
Timestamp: {self.timestamp.isoformat()}

--- Biomaterial Provenance ---
Material ID: {self.biomaterial_id}
Verification Status: {self.biomaterial_verification.status.value}
Credential Hash: {self.biomaterial_credential_hash[:16]}...

--- Drug Screening ---
Molecule ID: {self.molecule_id}
SMILES: {self.smiles[:50]}{'...' if len(self.smiles) > 50 else ''}
Model Hash: {self.model_hash[:16] if self.model_hash else 'N/A'}...

--- Blockchain Anchor ---
Master Hash: {self.master_hash[:16]}...
PureChain TX: {self.purechain_tx or 'Not yet anchored'}

Provenance Valid: {'YES' if self.is_valid_provenance() else 'NO'}
"""
