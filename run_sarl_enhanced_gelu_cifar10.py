#!/usr/bin/env python3
"""
Simple launcher script for Enhanced SARL with GELU on Sequential CIFAR-10
Usage: python run_sarl_enhanced_gelu_cifar10.py [mode]
Modes: quick, best, ablation, all
"""

import sys
import os

def main():
    """Main launcher function"""
    
    # Add the scripts directory to the path
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'sarl_enhanced_gelu', 'seq-cifar10.py')
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        print("Please make sure the script exists in scripts/sarl_enhanced_gelu/seq-cifar10.py")
        sys.exit(1)
    
    # Get mode from command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Enhanced SARL with GELU - Sequential CIFAR-10 Launcher")
        print("=" * 50)
        print("Available modes:")
        print("  quick    - Run a quick test (2 epochs, 1 seed)")
        print("  best     - Run best parameter experiments (50 epochs, 3 seeds)")
        print("  ablation - Run GELU ablation studies")
        print("  all      - Run all experiments")
        print()
        mode = input("Enter mode (quick/best/ablation/all): ").strip().lower()
    
    # Validate mode
    valid_modes = ['quick', 'best', 'ablation', 'all']
    if mode not in valid_modes:
        print(f"Invalid mode: {mode}")
        print(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)
    
    # Run the script
    print(f"Running sarl_enhanced_gelu experiments in {mode} mode...")
    print("=" * 50)
    
    # Import and run the main function from the script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'sarl_enhanced_gelu'))
    
    try:
        import seq_cifar10
        seq_cifar10.main()
    except ImportError as e:
        print(f"Error importing script: {e}")
        print("Trying to run as subprocess...")
        
        # Fallback to subprocess
        import subprocess
        cmd = [sys.executable, script_path, mode]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main() 
 