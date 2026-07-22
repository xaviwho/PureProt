#!/usr/bin/env python3
"""
A4 Part 1 -- Real validator failure injection on a geth v1.13.15 Clique testnet
(Reviewer R1.2).

Substrate: 4-signer go-ethereum v1.13.15 Clique PoA network (chain id
900520900520, period 2s) -- mirrors the real PureChain node observed on the live
RPC. This is a LOCAL TESTNET; it never contacts mainnet.

This measures the EMPIRICAL fault-tolerance envelope of Clique PoA:
  Phase 0  baseline           4/4 signers -- record block rate + rotation
  Phase 1  kill 1 signer      3/4 signers -- liveness under minority fault
  Phase 2  kill a 2nd signer  2/4 signers -- expected loss of majority -> stall
  Phase 3  restart both       recovery    -- time to resume + resync

Faults are injected with `docker kill` (abrupt SIGKILL = crash), recovery with
`docker start`. All block production is real; nothing is simulated. If a phase
produces zero blocks that is recorded as-is (a stall is a valid finding).

Outputs (results/a4_testnet/):
  failure_injection.json     -- structured results + recovery table
  failure_timeseries.csv     -- per-second block height samples across phases
  failure_injection_log.txt  -- raw run log (via tee by caller)

Run (testnet must already be up via the build steps):
  python -m blockchain.testnet.run_failure_injection
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

NODES = {1: 8545, 2: 8546, 3: 8547, 4: 8548}
CONTAINER = {i: f"pc-node{i}" for i in NODES}
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "a4_testnet",
)


def rpc(port, method, params=None):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": params or []}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.load(r).get("result")
    except Exception:
        return None


def block_height(port):
    h = rpc(port, "eth_blockNumber")
    return int(h, 16) if h else None


def live_ports():
    return [p for p in NODES.values() if block_height(p) is not None]


def max_height():
    hs = [block_height(p) for p in NODES.values()]
    hs = [h for h in hs if h is not None]
    return max(hs) if hs else None


def recent_signers(port=8545):
    """Return {blockNum:int -> signer} from clique snapshot recents (rotation)."""
    snap = rpc(port, "clique_getSnapshot")
    if not snap:
        return {}
    return {int(k): v for k, v in (snap.get("recents") or {}).items()}


def docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def sample_phase(label, duration_s, csv_rows, kill_action=None):
    """Sample max block height once per second for duration_s. Returns summary."""
    if kill_action:
        kill_action()
    t0 = time.time()
    start_h = max_height()
    signers_seen = set()
    samples = 0
    last_h = start_h
    while time.time() - t0 < duration_s:
        h = max_height()
        nlive = len(live_ports())
        csv_rows.append((label, round(time.time() - t0, 1), h if h is not None else "", nlive))
        if h is not None:
            last_h = h
        for s in recent_signers().values():
            signers_seen.add(s)
        samples += 1
        time.sleep(1.0)
    end_h = last_h
    produced = (end_h - start_h) if (start_h is not None and end_h is not None) else 0
    rate = produced / duration_s
    return {
        "phase": label,
        "duration_s": duration_s,
        "start_height": start_h,
        "end_height": end_h,
        "blocks_produced": produced,
        "block_rate_per_s": round(rate, 3),
        "live_nodes": len(live_ports()),
        "distinct_signers_observed": len(signers_seen),
        "signers_observed": sorted(signers_seen),
    }


def measure_recovery(restart_action, timeout_s=60):
    """Restart killed nodes; measure seconds until the chain produces a new block."""
    h_before = max_height()
    restart_action()
    t0 = time.time()
    first_new_block_s = None
    while time.time() - t0 < timeout_s:
        h = max_height()
        if h is not None and h_before is not None and h > h_before:
            first_new_block_s = round(time.time() - t0, 2)
            break
        time.sleep(0.5)
    return {
        "height_at_restart": h_before,
        "recovery_time_s": first_new_block_s,
        "recovered": first_new_block_s is not None,
        "height_after": max_height(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== A4 Part 1: Clique failure injection (real geth v1.13.15 testnet) ===")
    print(f"chain id 900520900520, 4 signers, period 2s")
    print(f"live nodes at start: {len(live_ports())}/4, height {max_height()}\n")

    csv_rows = []
    results = {"phases": []}

    # Phase 0: baseline 4/4
    print("Phase 0: baseline (4/4 signers), 20s ...")
    results["phases"].append(sample_phase("baseline_4of4", 20, csv_rows))

    # Phase 1: kill node2 -> 3/4
    def kill2():
        print("  >>> docker kill pc-node2 (abrupt crash)")
        docker("kill", CONTAINER[2])
    print("Phase 1: after killing 1 signer (3/4), 20s ...")
    results["phases"].append(sample_phase("kill1_3of4", 20, csv_rows, kill_action=kill2))

    # Phase 2: kill node3 -> 2/4
    def kill3():
        print("  >>> docker kill pc-node3 (abrupt crash)")
        docker("kill", CONTAINER[3])
    print("Phase 2: after killing a 2nd signer (2/4), 25s ...")
    results["phases"].append(sample_phase("kill2_2of4", 25, csv_rows, kill_action=kill3))

    # Phase 3: restart both -> recovery
    def restart():
        print("  >>> docker start pc-node2 pc-node3 (recovery)")
        docker("start", CONTAINER[2])
        docker("start", CONTAINER[3])
    print("Phase 3: recovery after restart ...")
    recovery = measure_recovery(restart)
    results["recovery"] = recovery
    # settle + measure restored rate
    time.sleep(8)
    print("Phase 3b: restored steady state (4/4), 20s ...")
    results["phases"].append(sample_phase("recovered_4of4", 20, csv_rows))

    # Write artifacts
    with open(os.path.join(OUT_DIR, "failure_timeseries.csv"), "w", newline="") as f:
        f.write("phase,t_s,max_height,live_nodes\n")
        for r in csv_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(OUT_DIR, "failure_injection.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print("\n--- SUMMARY (real measured block production) ---")
    print(f"{'phase':18}{'live':6}{'blocks':8}{'blk/s':8}{'signers':8}")
    for p in results["phases"]:
        print(f"{p['phase']:18}{p['live_nodes']:<6}{p['blocks_produced']:<8}"
              f"{p['block_rate_per_s']:<8}{p['distinct_signers_observed']:<8}")
    rec = results["recovery"]
    print(f"\nrecovery: recovered={rec['recovered']} "
          f"time={rec['recovery_time_s']}s "
          f"(height {rec['height_at_restart']} -> {rec['height_after']})")
    print(f"\nWrote results/a4_testnet/failure_injection.json + failure_timeseries.csv")


if __name__ == "__main__":
    main()
