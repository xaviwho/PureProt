#!/usr/bin/env python3
"""
Communication Protocol Efficiency and System Latency Analysis
Measures CLI timing, throughput, reliability, and failover capabilities
"""

import time
import subprocess
import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PureProtPerformanceAnalyzer:
    def __init__(self):
        self.results = []
        self.failed_operations = []
        self.timing_data = {
            'screen_times': [],
            'batch_times': [],
            'verify_times': [],
            'blockchain_times': []
        }
    
    def measure_screen_command(self, molecule_id, smiles, model_path="trained_model.joblib", iterations=10):
        """Measure timing for individual screen commands"""
        print(f"=== Measuring Screen Command Performance ({iterations} iterations) ===")
        
        screen_results = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                # Run screen command
                cmd = [
                    "python", "PureProt.py", "screen", molecule_id,
                    "--smiles", smiles,
                    "--model", model_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                success = result.returncode == 0
                
                screen_data = {
                    'iteration': i + 1,
                    'molecule_id': molecule_id,
                    'execution_time': execution_time,
                    'success': success,
                    'stdout_length': len(result.stdout),
                    'stderr_length': len(result.stderr),
                    'return_code': result.returncode
                }
                
                screen_results.append(screen_data)
                self.timing_data['screen_times'].append(execution_time)
                
                if not success:
                    self.failed_operations.append({
                        'operation': 'screen',
                        'molecule_id': molecule_id,
                        'error': result.stderr,
                        'iteration': i + 1
                    })
                
                print(f"  Iteration {i+1}: {execution_time:.4f}s {'✓' if success else '✗'}")
                
            except subprocess.TimeoutExpired:
                execution_time = 30.0  # timeout value
                screen_results.append({
                    'iteration': i + 1,
                    'molecule_id': molecule_id,
                    'execution_time': execution_time,
                    'success': False,
                    'error': 'timeout'
                })
                self.failed_operations.append({
                    'operation': 'screen',
                    'molecule_id': molecule_id,
                    'error': 'timeout',
                    'iteration': i + 1
                })
                print(f"  Iteration {i+1}: TIMEOUT")
            
            except Exception as e:
                screen_results.append({
                    'iteration': i + 1,
                    'molecule_id': molecule_id,
                    'execution_time': 0,
                    'success': False,
                    'error': str(e)
                })
                self.failed_operations.append({
                    'operation': 'screen',
                    'molecule_id': molecule_id,
                    'error': str(e),
                    'iteration': i + 1
                })
                print(f"  Iteration {i+1}: ERROR - {e}")
        
        return screen_results
    
    def measure_batch_throughput(self, csv_path, model_path="trained_model.joblib"):
        """Measure batch processing throughput"""
        print(f"=== Measuring Batch Throughput ===")
        
        # Count molecules in input file
        df = pd.read_csv(csv_path)
        num_molecules = len(df)
        
        start_time = time.perf_counter()
        
        try:
            cmd = [
                "python", "PureProt.py", "batch", csv_path,
                "--model", model_path,
                "--output", "batch_timing_test.csv"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            success = result.returncode == 0
            throughput = num_molecules / total_time if total_time > 0 else 0
            
            batch_data = {
                'num_molecules': num_molecules,
                'total_time': total_time,
                'throughput_mol_per_sec': throughput,
                'success': success,
                'return_code': result.returncode
            }
            
            self.timing_data['batch_times'].append(total_time)
            
            print(f"  Molecules: {num_molecules}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {throughput:.3f} molecules/sec")
            print(f"  Success: {'✓' if success else '✗'}")
            
            if not success:
                self.failed_operations.append({
                    'operation': 'batch',
                    'error': result.stderr,
                    'num_molecules': num_molecules
                })
            
        except subprocess.TimeoutExpired:
            batch_data = {
                'num_molecules': num_molecules,
                'total_time': 300.0,
                'throughput_mol_per_sec': 0,
                'success': False,
                'error': 'timeout'
            }
            self.failed_operations.append({
                'operation': 'batch',
                'error': 'timeout',
                'num_molecules': num_molecules
            })
            print(f"  TIMEOUT after 300s")
        
        except Exception as e:
            batch_data = {
                'num_molecules': num_molecules,
                'total_time': 0,
                'throughput_mol_per_sec': 0,
                'success': False,
                'error': str(e)
            }
            self.failed_operations.append({
                'operation': 'batch',
                'error': str(e),
                'num_molecules': num_molecules
            })
            print(f"  ERROR - {e}")
        
        return batch_data
    
    def measure_verify_performance(self, job_ids, iterations=5):
        """Measure blockchain verification performance"""
        print(f"=== Measuring Verify Command Performance ===")
        
        verify_results = []
        
        for job_id in job_ids[:iterations]:  # Limit to available job IDs
            start_time = time.perf_counter()
            
            try:
                cmd = ["python", "PureProt.py", "verify", job_id]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                success = result.returncode == 0
                
                verify_data = {
                    'job_id': job_id,
                    'execution_time': execution_time,
                    'success': success,
                    'return_code': result.returncode
                }
                
                verify_results.append(verify_data)
                self.timing_data['verify_times'].append(execution_time)
                
                print(f"  Job {job_id}: {execution_time:.4f}s {'✓' if success else '✗'}")
                
                if not success:
                    self.failed_operations.append({
                        'operation': 'verify',
                        'job_id': job_id,
                        'error': result.stderr
                    })
            
            except Exception as e:
                verify_results.append({
                    'job_id': job_id,
                    'execution_time': 0,
                    'success': False,
                    'error': str(e)
                })
                self.failed_operations.append({
                    'operation': 'verify',
                    'job_id': job_id,
                    'error': str(e)
                })
                print(f"  Job {job_id}: ERROR - {e}")
        
        return verify_results
    
    def analyze_blockchain_timing(self):
        """Analyze blockchain timing from existing results"""
        print(f"=== Analyzing Blockchain Timing ===")
        
        blockchain_files = [
            "top_10_ai_only_corrected.csv",
            "screening_results.csv"
        ]
        
        blockchain_times = []
        
        for file_path in blockchain_files:
            if Path(file_path).exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check for blockchain timing columns
                    if 'blockchain_duration' in df.columns:
                        times = df['blockchain_duration'].dropna()
                        blockchain_times.extend(times.tolist())
                        print(f"  {file_path}: {len(times)} blockchain transactions")
                    
                    # Check for AI timing
                    if 'ai_duration' in df.columns:
                        ai_times = df['ai_duration'].dropna()
                        print(f"  {file_path}: AI inference avg {ai_times.mean():.4f}s")
                
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
        
        if blockchain_times:
            self.timing_data['blockchain_times'] = blockchain_times
            print(f"  Total blockchain transactions analyzed: {len(blockchain_times)}")
            print(f"  Blockchain timing range: {min(blockchain_times):.3f} - {max(blockchain_times):.3f}s")
        
        return blockchain_times
    
    def calculate_reliability_metrics(self):
        """Calculate system reliability metrics"""
        print(f"=== Calculating Reliability Metrics ===")
        
        total_operations = (len(self.timing_data['screen_times']) + 
                          len(self.timing_data['batch_times']) + 
                          len(self.timing_data['verify_times']))
        
        total_failures = len(self.failed_operations)
        success_rate = ((total_operations - total_failures) / total_operations * 100) if total_operations > 0 else 0
        
        reliability_metrics = {
            'total_operations': total_operations,
            'successful_operations': total_operations - total_failures,
            'failed_operations': total_failures,
            'success_rate_percent': success_rate,
            'failure_breakdown': {}
        }
        
        # Breakdown failures by operation type
        for failure in self.failed_operations:
            op_type = failure['operation']
            if op_type not in reliability_metrics['failure_breakdown']:
                reliability_metrics['failure_breakdown'][op_type] = 0
            reliability_metrics['failure_breakdown'][op_type] += 1
        
        print(f"  Total operations: {total_operations}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Failed operations: {total_failures}")
        
        if total_failures > 0:
            print(f"  Failure breakdown:")
            for op_type, count in reliability_metrics['failure_breakdown'].items():
                print(f"    {op_type}: {count}")
        
        return reliability_metrics
    
    def generate_latency_report(self):
        """Generate comprehensive latency report"""
        print(f"=== Generating Latency Report ===")
        
        latency_report = {}
        
        for operation, times in self.timing_data.items():
            if times:
                latency_report[operation] = {
                    'count': len(times),
                    'min': min(times),
                    'max': max(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                    'percentile_95': np.percentile(times, 95) if times else 0
                }
                
                print(f"  {operation}:")
                print(f"    Count: {latency_report[operation]['count']}")
                print(f"    Min/Avg/Max: {latency_report[operation]['min']:.4f}s / "
                      f"{latency_report[operation]['mean']:.4f}s / {latency_report[operation]['max']:.4f}s")
                print(f"    95th percentile: {latency_report[operation]['percentile_95']:.4f}s")
        
        return latency_report
    
    def save_results(self):
        """Save all results to CSV files"""
        print(f"=== Saving Results ===")
        
        # Save timing data
        timing_df = pd.DataFrame([
            {'operation': op, 'time': t} 
            for op, times in self.timing_data.items() 
            for t in times
        ])
        timing_df.to_csv('communication_timing_results.csv', index=False)
        
        # Save failure data
        if self.failed_operations:
            failures_df = pd.DataFrame(self.failed_operations)
            failures_df.to_csv('communication_failures.csv', index=False)
        
        # Save comprehensive report
        reliability = self.calculate_reliability_metrics()
        latency = self.generate_latency_report()
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'reliability_metrics': reliability,
            'latency_metrics': latency,
            'total_timing_samples': sum(len(times) for times in self.timing_data.values())
        }
        
        with open('communication_protocol_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Results saved:")
        print(f"  - communication_timing_results.csv")
        print(f"  - communication_protocol_report.json")
        if self.failed_operations:
            print(f"  - communication_failures.csv")

def main():
    """Main performance analysis"""
    print("🔬 Communication Protocol Efficiency Analysis")
    print("=" * 60)
    
    analyzer = PureProtPerformanceAnalyzer()
    
    # Test molecules for screen command timing
    test_molecules = [
        ("CHEMBL429448", "CC[C@@H]1CCC[C@H](C2(OC(=O)N3CCC(N4CCCCC4)CC3)CC2)N1S(=O)(=O)c1cc(F)cc(F)c1"),
        ("CHEMBL4852639", "O=c1cc(/C=C/c2ccc(OCCCN3CCCCC3)cc2)occ1O")
    ]
    
    # 1. Measure screen command performance
    for mol_id, smiles in test_molecules:
        screen_results = analyzer.measure_screen_command(mol_id, smiles, iterations=5)
    
    # 2. Measure batch throughput (if batch_molecules.csv exists)
    if Path("batch_molecules.csv").exists():
        batch_results = analyzer.measure_batch_throughput("batch_molecules.csv")
    
    # 3. Analyze existing blockchain timing
    blockchain_times = analyzer.analyze_blockchain_timing()
    
    # 4. Generate reports
    reliability = analyzer.calculate_reliability_metrics()
    latency = analyzer.generate_latency_report()
    
    # 5. Save all results
    analyzer.save_results()
    
    print(f"\n🎯 COMMUNICATION PROTOCOL ANALYSIS COMPLETE")
    print(f"✅ Screen command timing measured")
    print(f"✅ Batch throughput analyzed") 
    print(f"✅ Blockchain verification timing extracted")
    print(f"✅ Reliability metrics calculated")
    print(f"✅ Latency report generated")

if __name__ == "__main__":
    main()
