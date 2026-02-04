// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title BioPassportCredentials
 * @dev Smart contract for managing biomaterial credentials on PureChain.
 * Part of the Context-Aware Drug Discovery system for ICUFN 2026.
 */
contract BioPassportCredentials {
    address public owner;

    // Credential types
    enum CredentialType { IDENTITY, QC_MYCO, TRANSFER, USAGE_RIGHTS }

    // Credential status
    enum CredentialStatus { ACTIVE, EXPIRED, REVOKED, QUARANTINED }

    // Biomaterial credential structure
    struct Credential {
        bytes32 credentialId;
        string materialId;          // e.g., "bio:cell_line:hela-001"
        CredentialType credType;
        CredentialStatus status;
        address issuer;
        uint256 issuedAt;
        uint256 expiresAt;
        bytes32 dataHash;           // Hash of credential data
    }

    // Material verification result
    struct VerificationResult {
        bool hasIdentity;
        bool hasQcMyco;
        bool notQuarantined;
        bool notRevoked;
        bool transferChainValid;
        bytes32 credentialHash;
        uint256 verifiedAt;
    }

    uint256 public credentialCount;

    // Mapping from material ID hash to credentials
    mapping(bytes32 => Credential[]) public materialCredentials;

    // Mapping from credential ID to credential
    mapping(bytes32 => Credential) public credentials;

    // Verification cache
    mapping(bytes32 => VerificationResult) public verificationCache;

    event CredentialIssued(bytes32 indexed credentialId, string materialId, CredentialType credType);
    event CredentialRevoked(bytes32 indexed credentialId);
    event MaterialVerified(bytes32 indexed materialIdHash, bool passed);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /**
     * @dev Issue a new credential for a biomaterial
     */
    function issueCredential(
        string memory materialId,
        CredentialType credType,
        uint256 validityDays,
        bytes32 dataHash
    ) public returns (bytes32) {
        bytes32 materialIdHash = keccak256(abi.encodePacked(materialId));
        bytes32 credentialId = keccak256(abi.encodePacked(
            materialId,
            credType,
            block.timestamp,
            msg.sender
        ));

        Credential memory newCred = Credential({
            credentialId: credentialId,
            materialId: materialId,
            credType: credType,
            status: CredentialStatus.ACTIVE,
            issuer: msg.sender,
            issuedAt: block.timestamp,
            expiresAt: block.timestamp + (validityDays * 1 days),
            dataHash: dataHash
        });

        credentials[credentialId] = newCred;
        materialCredentials[materialIdHash].push(newCred);
        credentialCount++;

        emit CredentialIssued(credentialId, materialId, credType);
        return credentialId;
    }

    /**
     * @dev Verify a biomaterial's credentials
     */
    function verifyMaterial(string memory materialId) public returns (VerificationResult memory) {
        bytes32 materialIdHash = keccak256(abi.encodePacked(materialId));
        Credential[] storage creds = materialCredentials[materialIdHash];

        VerificationResult memory result = VerificationResult({
            hasIdentity: false,
            hasQcMyco: false,
            notQuarantined: true,
            notRevoked: true,
            transferChainValid: true,
            credentialHash: bytes32(0),
            verifiedAt: block.timestamp
        });

        bytes32 combinedHash = bytes32(0);

        for (uint i = 0; i < creds.length; i++) {
            Credential storage cred = creds[i];

            // Check if credential is still valid
            if (cred.status == CredentialStatus.REVOKED) {
                result.notRevoked = false;
            }
            if (cred.status == CredentialStatus.QUARANTINED) {
                result.notQuarantined = false;
            }

            // Skip expired credentials
            if (block.timestamp > cred.expiresAt) {
                continue;
            }

            // Check credential types
            if (cred.credType == CredentialType.IDENTITY && cred.status == CredentialStatus.ACTIVE) {
                result.hasIdentity = true;
            }
            if (cred.credType == CredentialType.QC_MYCO && cred.status == CredentialStatus.ACTIVE) {
                result.hasQcMyco = true;
            }

            // Combine hashes
            combinedHash = keccak256(abi.encodePacked(combinedHash, cred.dataHash));
        }

        result.credentialHash = combinedHash;

        // Cache the result
        verificationCache[materialIdHash] = result;

        bool passed = result.hasIdentity && result.hasQcMyco &&
                      result.notQuarantined && result.notRevoked &&
                      result.transferChainValid;

        emit MaterialVerified(materialIdHash, passed);
        return result;
    }

    /**
     * @dev Get cached verification result
     */
    function getVerificationResult(string memory materialId) public view returns (VerificationResult memory) {
        bytes32 materialIdHash = keccak256(abi.encodePacked(materialId));
        return verificationCache[materialIdHash];
    }

    /**
     * @dev Revoke a credential
     */
    function revokeCredential(bytes32 credentialId) public onlyOwner {
        credentials[credentialId].status = CredentialStatus.REVOKED;
        emit CredentialRevoked(credentialId);
    }

    /**
     * @dev Get credential count for a material
     */
    function getMaterialCredentialCount(string memory materialId) public view returns (uint256) {
        bytes32 materialIdHash = keccak256(abi.encodePacked(materialId));
        return materialCredentials[materialIdHash].length;
    }
}
