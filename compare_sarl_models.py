#!/usr/bin/env python3
"""
Comparison script for SARL models
Tests both original SARL and enhanced SARL with contrastive-based semantic learning
"""

import os
import subprocess
import time
from datetime import datetime

def run_experiment(model_name, dataset, buffer_size, seed, output_dir):
    """Run a single experiment"""
    
    if model_name == "sarl":
        # Original SARL parameters
        params = {
            'alpha': 0.5,
            'beta': 1,
            'op_weight': 0.5,
            'sim_thresh': 0.8,
            'sm_weight': 0.01,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        }
    elif model_name == "sarl_enhanced":
        # Enhanced SARL parameters
        params = {
            'alpha': 0.5,
            'beta': 1,
            'op_weight': 0.5,
            'sm_weight': 0.01,
            'sim_lr': 0.001,
            'kw': '0.9 0.9 0.9 0.9',
            'lr': 0.03,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 50,
            'lr_steps': '35 45',
            'warmup_epochs': 3
        }
    
    exp_id = f"{model_name}-{dataset}-{buffer_size}-s-{seed}"
    
    cmd = [
        "python", "main.py",
        "--experiment_id", exp_id,
        "--model", model_name,
        "--dataset", dataset,
        "--kw", params['kw'],
        "--alpha", str(params['alpha']),
        "--beta", str(params['beta']),
        "--op_weight", str(params['op_weight']),
        "--sm_weight", str(params['sm_weight']),
        "--buffer_size", str(buffer_size),
        "--batch_size", str(params['batch_size']),
        "--minibatch_size", str(params['minibatch_size']),
        "--lr", str(params['lr']),
        "--lr_steps", params['lr_steps'],
        "--n_epochs", str(params['n_epochs']),
        "--output_dir", output_dir,
        "--csv_log",
        "--seed", str(seed),
        "--device", "cuda",
        "--save_model", "0",
        "--save_interim", "0"
    ]
    
    # Add model-specific parameters
    if model_name == "sarl":
        cmd.extend(["--sim_thresh", str(params['sim_thresh'])])
    elif model_name == "sarl_enhanced":
        cmd.extend(["--sim_lr", str(params['sim_lr'])])
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    return {
        'model': model_name,
        'dataset': dataset,
        'buffer_size': buffer_size,
        'seed': seed,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration': end_time - start_time
    }

def main():
    """Main comparison function"""
    
    # Experiment configuration
    models = ["sarl", "sarl_enhanced"]
    datasets = ["seq-cifar100"]
    buffer_sizes = [200, 500]
    seeds = [1, 3, 5]
    
    output_dir = "experiments/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    print("=" * 80)
    print("SARL Model Comparison")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Buffer sizes: {buffer_sizes}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    total_experiments = len(models) * len(datasets) * len(buffer_sizes) * len(seeds)
    current_experiment = 0
    
    for model in models:
        for dataset in datasets:
            for buffer_size in buffer_sizes:
                for seed in seeds:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}] Running {model} on {dataset} with buffer_size={buffer_size}, seed={seed}")
                    
                    result = run_experiment(model, dataset, buffer_size, seed, output_dir)
                    results.append(result)
                    
                    if result['return_code'] == 0:
                        print(f"✓ Success - Duration: {result['duration']:.2f}s")
                    else:
                        print(f"✗ Failed - Return code: {result['return_code']}")
                        print(f"Error: {result['stderr']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['return_code'] == 0]
    failed = [r for r in results if r['return_code'] != 0]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
        print(f"Average duration: {avg_duration:.2f}s")
    
    # Model-wise summary
    for model in models:
        model_results = [r for r in successful if r['model'] == model]
        if model_results:
            avg_duration = sum(r['duration'] for r in model_results) / len(model_results)
            print(f"{model}: {len(model_results)} successful, avg duration: {avg_duration:.2f}s")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"comparison_results_{timestamp}.txt")
    
    with open(results_file, 'w') as f:
        f.write("SARL Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Buffer size: {result['buffer_size']}\n")
            f.write(f"Seed: {result['seed']}\n")
            f.write(f"Success: {result['return_code'] == 0}\n")
            f.write(f"Duration: {result['duration']:.2f}s\n")
            if result['return_code'] != 0:
                f.write(f"Error: {result['stderr']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main() 