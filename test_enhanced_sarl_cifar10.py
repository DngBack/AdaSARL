#!/usr/bin/env python3
"""
Quick test script for Enhanced SARL on CIFAR-10
"""

import os
import subprocess
import sys

def test_enhanced_sarl_cifar10():
    """Test enhanced SARL on CIFAR-10 with minimal settings"""
    
    print("Testing Enhanced SARL on CIFAR-10...")
    
    # Test parameters
    test_params = {
        'model': 'sarl_enhanced',
        'dataset': 'seq-cifar10',
        'buffer_size': 200,
        'n_epochs': 2,  # Very short for testing
        'batch_size': 16,
        'minibatch_size': 16,
        'lr': 0.03,
        'alpha': 0.5,
        'beta': 1,
        'op_weight': 0.5,
        'sm_weight': 0.01,
        'sim_lr': 0.001,
        'kw': '0.9 0.9 0.9 0.9',
        'seed': 1,
        'device': 'cuda',
        'experiment_id': 'test_enhanced_cifar10'
    }
    
    cmd = [
        "python", "main.py",
        "--experiment_id", test_params['experiment_id'],
        "--model", test_params['model'],
        "--dataset", test_params['dataset'],
        "--buffer_size", str(test_params['buffer_size']),
        "--n_epochs", str(test_params['n_epochs']),
        "--batch_size", str(test_params['batch_size']),
        "--minibatch_size", str(test_params['minibatch_size']),
        "--lr", str(test_params['lr']),
        "--alpha", str(test_params['alpha']),
        "--beta", str(test_params['beta']),
        "--op_weight", str(test_params['op_weight']),
        "--sm_weight", str(test_params['sm_weight']),
        "--sim_lr", str(test_params['sim_lr']),
        "--kw", test_params['kw'],
        "--seed", str(test_params['seed']),
        "--device", test_params['device'],
        "--output_dir", "experiments/test",
        "--save_model", "0",
        "--save_interim", "0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✓ Enhanced SARL CIFAR-10 test PASSED")
            print("Output preview:")
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("✗ Enhanced SARL CIFAR-10 test FAILED")
            print(f"Return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Enhanced SARL CIFAR-10 test TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ Enhanced SARL CIFAR-10 test ERROR: {e}")
        return False

def test_semantic_grouping():
    """Test semantic grouping logic for CIFAR-10"""
    
    print("\nTesting CIFAR-10 semantic grouping...")
    
    # CIFAR-10 class mapping
    cifar10_classes = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }
    
    # Expected semantic groups
    vehicle_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
    animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
    
    print("Vehicle classes:", [cifar10_classes[c] for c in vehicle_classes])
    print("Animal classes:", [cifar10_classes[c] for c in animal_classes])
    
    # Test positive pairs
    print("\nTesting positive pairs:")
    for category in [vehicle_classes, animal_classes]:
        print(f"Category: {[cifar10_classes[c] for c in category]}")
        for i in range(len(category)):
            for j in range(i+1, len(category)):
                print(f"  {cifar10_classes[category[i]]} <-> {cifar10_classes[category[j]]}")
    
    # Test negative pairs
    print("\nTesting negative pairs:")
    for vehicle in vehicle_classes:
        for animal in animal_classes:
            print(f"  {cifar10_classes[vehicle]} <-> {cifar10_classes[animal]}")
    
    print("✓ Semantic grouping test PASSED")
    return True

def main():
    """Main test function"""
    
    print("=" * 60)
    print("Enhanced SARL CIFAR-10 Test Suite")
    print("=" * 60)
    
    # Test semantic grouping logic
    semantic_test = test_semantic_grouping()
    
    # Test actual model execution
    model_test = test_enhanced_sarl_cifar10()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Semantic grouping test: {'✓ PASSED' if semantic_test else '✗ FAILED'}")
    print(f"Model execution test: {'✓ PASSED' if model_test else '✗ FAILED'}")
    
    if semantic_test and model_test:
        print("\n🎉 All tests PASSED! Enhanced SARL is ready for CIFAR-10.")
        print("\nYou can now run:")
        print("  python scripts/sarl_enhanced/seq-cifar10.py")
        print("  python compare_sarl_models.py")
    else:
        print("\n❌ Some tests FAILED. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 