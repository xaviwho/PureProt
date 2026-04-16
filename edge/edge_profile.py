#!/usr/bin/env python3
"""
PureProtX Edge Node Profiling

Simulates running the PureProtX pipeline under resource-constrained edge
hardware by capping Docker container CPU and memory.  Measures latency,
peak memory, and throughput at each tier.

Four hardware tiers:
  server       -- no caps (baseline)
  rpi4         -- 4 CPU cores, 4 GB RAM  (Raspberry Pi 4)
  jetson_nano  -- 4 CPU cores, 2 GB RAM  (NVIDIA Jetson Nano)
  constrained  -- 1 CPU core,  512 MB RAM (minimal sensor gateway)

Dependencies: docker>=7.0.0, pyyaml

Output:
  results/edge_benchmark.csv
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
PROFILE_NAMES = ["server", "rpi4", "jetson_nano", "constrained"]

CONTAINER_TIMEOUT_S = 600  # 10 minute hard timeout per run


# ------------------------------------------------------------------
# Profile loading
# ------------------------------------------------------------------

def load_profile(profile_name: str) -> Dict[str, Any]:
    """Load a hardware profile YAML file."""
    yaml_path = os.path.join(PROFILES_DIR, f"{profile_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Profile not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Docker-based pipeline execution
# ------------------------------------------------------------------

def _import_docker_sdk():
    """
    Import the docker Python SDK while bypassing the local ./docker/
    directory, which otherwise shadows the package because PROJECT_ROOT
    sits at the front of sys.path.
    """
    import importlib
    saved_path = sys.path[:]
    sys.path = [p for p in sys.path if os.path.abspath(p) != PROJECT_ROOT]
    try:
        # If the local namespace package was already imported, drop it
        if "docker" in sys.modules and not hasattr(sys.modules["docker"], "from_env"):
            del sys.modules["docker"]
        return importlib.import_module("docker")
    finally:
        sys.path = saved_path


def _docker_available() -> bool:
    """Check if Docker SDK is importable and the daemon is running."""
    try:
        docker = _import_docker_sdk()
        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        logger.debug("Docker check failed: %s", e)
        return False


def profile_pipeline(
    profile_name: str,
    target: str,
    n_compounds: int,
    image: str = "pureprot-edge:latest",
) -> Dict[str, Any]:
    """
    Runs ``pureprot screen --target {target} --n {n_compounds}``
    inside a resource-capped Docker container.

    Returns:
      {
        "profile": str,
        "target": str,
        "n_compounds": int,
        "latency_s": float,
        "peak_mem_mb": float,
        "throughput_cpm": float,      # compounds per minute
        "blockchain_latency_ms": float,
        "success": bool
      }
    """
    profile = load_profile(profile_name)

    result = {
        "profile": profile_name,
        "target": target,
        "n_compounds": n_compounds,
        "latency_s": 0.0,
        "peak_mem_mb": 0.0,
        "throughput_cpm": 0.0,
        "blockchain_latency_ms": 0.0,
        "success": False,
    }

    if not _docker_available():
        logger.info("Docker unavailable -- running offline simulation for %s", profile_name)
        return _simulate_profile(profile_name, profile, target, n_compounds)

    try:
        docker = _import_docker_sdk()
        client = docker.from_env()

        # Build container resource constraints.
        # On Windows Docker Desktop / WSL2, cpu_quota is not always enforced;
        # cpuset_cpus (which pins the container to specific CPU cores) IS
        # enforced by the kernel scheduler regardless of host. We compute
        # the cpuset string from cpu_quota for portable behaviour.
        kwargs = {}
        cpu_quota = profile.get("cpu_quota", 0)
        cpu_period = profile.get("cpu_period", 100000)
        mem_limit = profile.get("mem_limit", "0")

        if cpu_quota and cpu_quota > 0:
            n_cpus = max(1, cpu_quota // cpu_period)
            kwargs["cpuset_cpus"] = ",".join(str(i) for i in range(n_cpus))
            kwargs["cpu_quota"] = cpu_quota
            kwargs["cpu_period"] = cpu_period
        if mem_limit and mem_limit != "0":
            kwargs["mem_limit"] = mem_limit
            kwargs["memswap_limit"] = mem_limit  # disable swap, force OOM at limit

        # The pureprot-edge image takes args directly via ENTRYPOINT
        command = ["--target", target, "--n", str(n_compounds)]

        t0 = time.perf_counter()
        container = client.containers.run(
            image,
            command=command,
            detach=True,
            **kwargs,
        )

        # Wait for completion with timeout
        exit_result = container.wait(timeout=CONTAINER_TIMEOUT_S)
        latency = time.perf_counter() - t0

        exit_code = exit_result.get("StatusCode", -1)

        # Parse the EDGE_RESULT_JSON line from the container's stdout.
        # This carries the workload's own internal measurements
        # (peak mem from /sys/fs/cgroup, inference time, blockchain latency).
        logs = container.logs().decode("utf-8", errors="replace")
        workload = _parse_edge_result_json(logs)

        # Prefer cgroup-reported peak memory; fall back to docker stats
        peak_mem_mb = workload.get("peak_mem_mb", 0.0)
        if peak_mem_mb == 0.0:
            try:
                stats = container.stats(stream=False)
                peak_mem_bytes = stats.get("memory_stats", {}).get("max_usage", 0)
                peak_mem_mb = peak_mem_bytes / (1024 * 1024)
            except Exception:
                pass

        bc_latency = workload.get("blockchain_latency_ms", 0.0)
        bc_success = workload.get("blockchain_success", False)
        block_number = workload.get("block_number", 0)

        container.remove(force=True)

        throughput = workload.get("throughput_cpm", (n_compounds / latency) * 60 if latency > 0 else 0.0)

        result.update({
            "latency_s": round(latency, 3),
            "peak_mem_mb": round(peak_mem_mb, 2),
            "throughput_cpm": round(throughput, 1),
            "blockchain_latency_ms": round(bc_latency, 2),
            "blockchain_success": bc_success,
            "block_number": block_number,
            "success": exit_code == 0,
        })

    except Exception as e:
        logger.error("Docker profiling failed for %s/%s/n=%d: %s",
                     profile_name, target, n_compounds, e)
        # Record the failure gracefully rather than crashing
        result["error"] = str(e)

    return result


def _parse_edge_result_json(logs: str) -> Dict[str, Any]:
    """
    Parse the EDGE_RESULT_JSON line emitted by edge_workload.py.

    Returns the parsed dict, or {} if no such line was found.
    """
    for line in logs.splitlines():
        line = line.strip()
        if line.startswith("EDGE_RESULT_JSON:"):
            try:
                return json.loads(line.split(":", 1)[1].strip())
            except Exception:
                pass
    return {}


# ------------------------------------------------------------------
# Offline simulation (when Docker is unavailable)
# ------------------------------------------------------------------

def _simulate_profile(
    profile_name: str,
    profile: Dict[str, Any],
    target: str,
    n_compounds: int,
) -> Dict[str, Any]:
    """
    Simulate edge profiling without Docker.

    Uses empirical scaling factors derived from:
    - CPU: inverse proportional to cpu_quota (fewer cores = slower)
    - Memory: constrained profiles get penalty for potential swapping
    - Base latency from server profile (measured or estimated)
    """
    rng = np.random.RandomState(hash((profile_name, target, n_compounds)) % 2**31)

    # Baseline: ~0.03s per compound on server (unconstrained)
    base_per_compound_s = 0.030

    # CPU scaling factor
    cpu_quota = profile.get("cpu_quota", 0)
    if cpu_quota <= 0:
        cpu_factor = 1.0  # server: no cap
    else:
        n_cores = cpu_quota / profile.get("cpu_period", 100000)
        # Assume server has ~8 effective cores
        cpu_factor = 8.0 / n_cores

    # Memory scaling factor
    mem_str = profile.get("mem_limit", "0")
    if mem_str == "0":
        mem_factor = 1.0
    else:
        mem_mb = _parse_mem_mb(mem_str)
        if mem_mb < 1024:
            mem_factor = 2.5  # severe constraint -> swapping/OOM risk
        elif mem_mb < 3000:
            mem_factor = 1.3  # moderate
        else:
            mem_factor = 1.0

    per_compound_s = base_per_compound_s * cpu_factor * mem_factor
    latency_s = per_compound_s * n_compounds
    # Add random noise +/- 5%
    latency_s *= 1.0 + rng.uniform(-0.05, 0.05)

    # Estimate peak memory (MB) -- scales with compound count
    base_mem_mb = 80 + n_compounds * 0.05  # ~50 bytes features per compound + overhead
    peak_mem_mb = base_mem_mb * (1 + rng.uniform(0, 0.1))

    # Check for OOM on constrained profile
    mem_limit_mb = _parse_mem_mb(mem_str) if mem_str != "0" else float("inf")
    success = peak_mem_mb < mem_limit_mb

    if not success and profile_name == "constrained" and n_compounds > 1000:
        # Likely OOM on 512MB with >1000 compounds
        latency_s = 0.0
        peak_mem_mb = mem_limit_mb

    throughput = (n_compounds / latency_s) * 60 if latency_s > 0 else 0.0
    bc_latency = rng.uniform(15, 50)  # simulated PureChain commit

    return {
        "profile": profile_name,
        "target": target,
        "n_compounds": n_compounds,
        "latency_s": round(latency_s, 3),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "throughput_cpm": round(throughput, 1),
        "blockchain_latency_ms": round(bc_latency, 2),
        "success": success,
        "simulated": True,
    }


def _parse_mem_mb(mem_str: str) -> float:
    """Parse Docker memory limit string like '4g', '512m' to MB."""
    mem_str = str(mem_str).strip().lower()
    if mem_str.endswith("g"):
        return float(mem_str[:-1]) * 1024
    elif mem_str.endswith("m"):
        return float(mem_str[:-1])
    elif mem_str.endswith("k"):
        return float(mem_str[:-1]) / 1024
    else:
        try:
            # Assume bytes
            return float(mem_str) / (1024 * 1024)
        except ValueError:
            return 0.0


# ------------------------------------------------------------------
# Full benchmark
# ------------------------------------------------------------------

def run_full_edge_benchmark(
    targets: List[str] = None,
    batch_sizes: List[int] = None,
) -> pd.DataFrame:
    """
    Runs profile_pipeline for every combination of
    (profile x target x batch_size) and returns results as a DataFrame.
    Saves CSV to results/edge_benchmark.csv
    """
    if targets is None:
        targets = ["CHEMBL243"]
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000]

    rows = []
    total = len(PROFILE_NAMES) * len(targets) * len(batch_sizes)
    count = 0

    for profile_name in PROFILE_NAMES:
        for target in targets:
            for n in batch_sizes:
                count += 1
                print(f"  [{count}/{total}] {profile_name} / {target} / n={n} ...",
                      end="", flush=True)
                result = profile_pipeline(profile_name, target, n)
                rows.append(result)
                status = "OK" if result["success"] else "FAIL"
                print(f"  {status} ({result['latency_s']:.1f}s, "
                      f"{result['peak_mem_mb']:.0f}MB, "
                      f"{result['throughput_cpm']:.0f} cpm)")

    df = pd.DataFrame(rows)

    # Save CSV (append to existing if present, avoiding duplicate rows)
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "edge_benchmark.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # Drop rows with same (profile, target, n_compounds) to avoid dupes
        key_cols = ["profile", "target", "n_compounds"]
        new_keys = set(df[key_cols].apply(tuple, axis=1))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(new_keys)]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path} ({len(df)} total rows)")

    return df


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="CHEMBL243",
                        help="Comma-separated target IDs")
    parser.add_argument("--batch-sizes", default="100,500,1000",
                        help="Comma-separated compound counts")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",")]
    batch_sizes = [int(n.strip()) for n in args.batch_sizes.split(",")]

    print("=" * 60)
    print("PureProtX Edge Node Profiling Benchmark")
    print("=" * 60)

    df = run_full_edge_benchmark(targets=targets, batch_sizes=batch_sizes)
    print("\nResults:")
    print(df.to_string(index=False))
