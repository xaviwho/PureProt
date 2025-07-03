// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title DrugScreeningVerifier
 * @dev Smart contract for verifying drug screening results on Purechain.
 * This version uses a numeric ID to retrieve results to avoid node issues with bytes32 keys.
 */
contract DrugScreeningVerifier {
    address public owner;

    // The core data structure for a screening result
    struct ScreeningResult {
        bytes32 resultId; // The unique, verifiable hash of the result
        address researcher;
        uint256 timestamp;
        string moleculeId; // e.g., 'aspirin', 'ibuprofen'
        bytes32 moleculeDataHash; // Hash of the raw molecule data (SMILES, etc.)
        bytes32 resultHash; // Hash of the full result data
        bool verified;
    }

    uint256 public resultCount;
    // Mapping from a simple numeric ID to the result data
    mapping(uint256 => ScreeningResult) public screeningResults;

    // Event emitted when a new result is recorded
    event ResultRecorded(uint256 indexed numericId, bytes32 indexed resultId, address indexed researcher);
    event ResultVerified(uint256 indexed numericId, bool verified);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /**
     * @dev Records a new screening result on the blockchain.
     * @param resultHash The keccak256 hash of the screening result data
     * @param moleculeDataHash The keccak256 hash of the input molecule data
     * @param moleculeId A human-readable identifier for the molecule
     */
    function recordScreeningResult(bytes32 resultHash, bytes32 moleculeDataHash, string memory moleculeId) public {
        // Calculate the verifiable, content-based hash for the result
        bytes32 resultId = keccak256(abi.encodePacked(resultHash, msg.sender, block.timestamp));

        // Store the new result using a simple numeric ID
        screeningResults[resultCount] = ScreeningResult({
            resultId: resultId,
            researcher: msg.sender,
            timestamp: block.timestamp,
            moleculeId: moleculeId,
            moleculeDataHash: moleculeDataHash,
            resultHash: resultHash,
            verified: false
        });

        // Emit an event for off-chain listeners with both the numeric and hash-based IDs
        emit ResultRecorded(resultCount, resultId, msg.sender);

        // Increment the counter for the next result
        resultCount++;
    }

    /**
     * @dev Verifies a screening result by its numeric ID.
     * @param numericId The numeric ID of the result to verify.
     * @param verified The verification status to set.
     */
    function verifyResult(uint256 numericId, bool verified) public onlyOwner {
        require(numericId < resultCount, "Result ID out of bounds");
        screeningResults[numericId].verified = verified;
        emit ResultVerified(numericId, verified);
    }

    /**
     * @dev Retrieves a full screening result by its numeric ID.
     * @param numericId The ID of the result to retrieve.
     * @return The ScreeningResult struct.
     */
    function getScreeningResult(uint256 numericId) public view returns (ScreeningResult memory) {
        require(numericId < resultCount, "Result ID out of bounds");
        return screeningResults[numericId];
    }
}
