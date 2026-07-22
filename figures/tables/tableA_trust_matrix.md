| Mechanism | Single point of compromise | Rewrite resistance | Ordering \& timestamp | Public verification | Write latency (ms/rec) |
|---|---|---|---|---|---|
| Ed25519 signed log | The private key | Key holder can re-sign history | Yes (chained) | Needs trusted public key | 0.0523 |
| IPFS content addressing | The CID publisher | None on its own (no ordering) | No (content-addressed only) | Needs trusted CID index | 0.0074 |
| Merkle batch | The anchor | Needs an external anchor | No (unordered set) | Needs the anchor | 0.0011 |
| PoA² (PureChain) | None (distributed) | Requires majority-of-validators collusion (no single party) | Yes (block height + time) | Public tx/state read via RPC (validator-set query disabled on current public endpoint) | 1,880 (indiv.) / 20 (batch) |