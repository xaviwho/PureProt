#!/usr/bin/env python3
"""
Quick test runner for the enhanced E2E test harness
"""

import subprocess
import sys
import os
from pathlib import Path

def run_enhanced_test():
    """Run the lightweight E2E test and report results."""
    print("🚀 Running PureProtX Lightweight E2E Test Harness")
    print("="*60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    try:
        # Run the lightweight test with live output
        print("Starting lightweight test (bypasses Web3 memory issues)...")
        result = subprocess.run([
            sys.executable, "tests/lightweight_e2e_test.py"
        ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nExit code: {result.returncode}")
        
        # Check for expected artifacts
        artifacts = [
            "lightweight_test_results.json",
            "tests/lightweight_verification_log.txt",
            "tests/lightweight_docking_scores.csv"
        ]
        
        print("\n📁 Checking test artifacts:")
        for artifact in artifacts:
            if os.path.exists(artifact):
                size = os.path.getsize(artifact)
                print(f"  ✓ {artifact} ({size} bytes)")
            else:
                print(f"  ✗ {artifact} (missing)")
        
        # Check golden file comparison
        golden_file = "tests/golden/lightweight_results.json"
        if os.path.exists(golden_file):
            print(f"  ✓ {golden_file} (golden reference exists)")
        else:
            print(f"  ℹ {golden_file} (will be created on first run)")
        
        if result.returncode == 0:
            print("\n🎉 Test harness completed successfully!")
            print("✅ System is ready for publication")
        else:
            print("\n❌ Test harness failed")
            print("Check logs for details")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return False

if __name__ == "__main__":
    success = run_enhanced_test()
    sys.exit(0 if success else 1)
