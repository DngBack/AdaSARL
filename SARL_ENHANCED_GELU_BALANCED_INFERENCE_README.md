# SARL Enhanced GELU with Balanced Sampling and Inference-Only Prototypes

This is a new implementation that combines the best features from multiple SARL variants:

1. **GELU Activation Functions** from `sarl_enhanced_gelu_balanced.py`
2. **Balanced Sampling** from `sarl_enhanced_gelu_balanced.py`
3. **Inference-Only Prototypes** from `sarl_improve.py`

## Key Features

### 1. GELU Activation Functions

- Uses GELU (Gaussian Error Linear Unit) instead of ReLU
- Better gradient flow and performance
- Configurable application to different layers

### 2. Balanced Sampling

- **Semantic Similarity Network**: Learns to compute similarity between features
- **Adaptive Threshold**: Learnable threshold for semantic grouping
- **Balanced Buffer**: Ensures balanced sampling from different semantic groups
- **No Domain Knowledge**: Uses learned similarity instead of predefined class relationships

### 3. Inference-Only Prototypes (CRITICAL FIX)

- **No prototype guidance during training**: This was the main issue in original SARL
- **Conservative inference guidance**: Only 5% influence during evaluation
- **High momentum updates**: 0.99 momentum for stable prototype evolution
- **Simple single prototypes**: No complex multi-centroid clustering

## Model Architecture

### Core Components

1. **SARLEnhancedGeluBalancedInference**: Main model class
2. **SemanticSimilarity**: Learnable similarity network
3. **AdaptiveThreshold**: Learnable threshold for grouping
4. **BalancedBuffer**: Enhanced buffer with balanced sampling

### Key Methods

- `forward()`: **FIXED** - Only uses prototypes during inference with minimal weight
- `observe()`: **FIXED** - No prototype guidance during training
- `end_epoch()`: **FIXED** - High momentum prototype updates
- `end_task()`: **FIXED** - Simple, stable prototype calculation

## Usage

### Quick Start

```bash
# Run the complete training script
python run_sarl_enhanced_gelu_balanced_inference_cifar10.py
```

### Using the Script

```bash
# Run the standard training script
python scripts/sarl_enhanced_gelu_balanced_inference/seq-cifar10.py
```

### Key Parameters

```python
# Prototype parameters (inference-only)
prototype_momentum=0.99,        # High momentum for stability
enable_inference_guidance=1,    # Enable prototype guidance during inference
guidance_weight=0.05,          # Very small weight (5% influence)

# Balanced sampling
use_balanced_sampling=1,       # Enable balanced sampling
balance_weight=1.0,            # Weight for balance factor

# GELU
apply_gelu=[1, 1, 1, 1],       # Apply GELU to all layers
num_feats=512,                 # Feature dimension

# SARL specific
alpha=0.5,                     # Consistency regularization weight
beta=1.0,                      # Knowledge distillation weight
op_weight=0.1,                 # Prototype regularization weight
sm_weight=0.01,                # Semantic contrastive weight
```

## Key Improvements Over Original SARL

### 1. Fixed Prototype Issues

- **Before**: Prototypes used during training causing interference
- **After**: Prototypes only used during inference with minimal weight

### 2. Enhanced Semantic Learning

- **Before**: Used domain knowledge for class relationships
- **After**: Learns semantic similarity from data

### 3. Balanced Sampling

- **Before**: Uniform sampling from buffer
- **After**: Balanced sampling based on semantic groups

### 4. GELU Activation

- **Before**: ReLU activation
- **After**: GELU activation for better performance

## File Structure

```
models/
├── sarl_enhanced_gelu_balanced_inference.py    # Main model
scripts/
├── sarl_enhanced_gelu_balanced_inference/
│   └── seq-cifar10.py                         # Training script
run_sarl_enhanced_gelu_balanced_inference_cifar10.py  # Complete test script
```

## Expected Performance

This model should show:

- **Better stability**: No prototype interference during training
- **Improved accuracy**: GELU activation and balanced sampling
- **Reduced forgetting**: Better semantic preservation
- **More robust**: Conservative prototype guidance

## Comparison with Other Models

| Model              | Prototype Usage      | Activation | Sampling     | Domain Knowledge |
| ------------------ | -------------------- | ---------- | ------------ | ---------------- |
| Original SARL      | Training + Inference | ReLU       | Uniform      | Yes              |
| SARL Improve       | Inference Only       | ReLU       | Uniform      | Yes              |
| SARL Enhanced GELU | Training + Inference | GELU       | Balanced     | No               |
| **This Model**     | **Inference Only**   | **GELU**   | **Balanced** | **No**           |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or buffer size
2. **NaN losses**: Check learning rates and gradient clipping
3. **Poor performance**: Adjust prototype momentum and guidance weight

### Debugging

Enable detailed logging:

```python
args.tensorboard = True
args.csv_log = True
```

Check prototype evolution:

```python
# In the model, prototypes are saved after each task
# Check: outputs/sarl_enhanced_gelu_balanced_inference/task_models/
```

## Citation

If you use this model, please cite the original SARL paper and mention the improvements:

```bibtex
@article{sarl_original,
  title={Semantic-Aware Representation Learning for Continual Learning},
  author={...},
  journal={...},
  year={...}
}
```

## License

This implementation follows the same license as the original SARL codebase.
