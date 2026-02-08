#!/usr/bin/env python3
"""
Script to analyze training iterations and compute average metrics.
"""

import re
import argparse

def parse_log_file(log_file, start_iter=45, end_iter=95):
    """Parse log file and extract metrics for specified iteration range."""
    
    elapsed_times = []
    throughputs = []
    
    # Pattern to match iteration lines
    pattern = r'iteration\s+(\d+)/\s+\d+.*?elapsed time per iteration \(ms\):\s+([\d.]+).*?throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                elapsed_time = float(match.group(2))
                throughput = float(match.group(3))
                
                if start_iter <= iteration <= end_iter:
                    elapsed_times.append(elapsed_time)
                    throughputs.append(throughput)
    
    return elapsed_times, throughputs

def main():
    parser = argparse.ArgumentParser(
        description='Analyze training log iterations and compute average metrics.'
    )
    parser.add_argument(
        'log_file',
        help='Path to the log file to analyze'
    )
    parser.add_argument(
        '--start-iter',
        type=int,
        default=50,
        help='Start iteration (default: 55)'
    )
    parser.add_argument(
        '--end-iter',
        type=int,
        default=100,
        help='End iteration (default: 95)'
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing iterations {args.start_iter}-{args.end_iter} from {args.log_file}")
    print("=" * 70)
    
    elapsed_times, throughputs = parse_log_file(args.log_file, args.start_iter, args.end_iter)
    
    if not elapsed_times:
        print("No data found for the specified iteration range.")
        return
    
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    avg_throughput = sum(throughputs) / len(throughputs)
    
    print(f"\nNumber of iterations analyzed: {len(elapsed_times)}")
    print(f"\nAverage elapsed time per iteration: {avg_elapsed_time:.2f} ms")
    print(f"Average throughput per GPU: {avg_throughput:.2f} TFLOP/s/GPU")
    print(f"\nMin elapsed time: {min(elapsed_times):.2f} ms")
    print(f"Max elapsed time: {max(elapsed_times):.2f} ms")
    print(f"\nMin throughput: {min(throughputs):.2f} TFLOP/s/GPU")
    print(f"Max throughput: {max(throughputs):.2f} TFLOP/s/GPU")

if __name__ == "__main__":
    main()
