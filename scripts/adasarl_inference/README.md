# SARL Enhanced GELU Balanced Inference Scripts

This directory contains scripts for running experiments with the `sarl_enhanced_gelu_balanced_inference` model, which is an enhanced version of SARL with:

- **GELU Activation**: Replaces ReLU with GELU activation functions
- **Balanced Instance Sampling**: Semantic-aware buffer sampling for better representation
- **Inference-Only Prototypes**: Prototype guidance only during evaluation, not training

## Key Features

### Inference-Only Prototype Guidance

- Prototypes are only used during evaluation (`model.eval()` mode)
- Very conservative guidance weight (0.05-0.1) to prevent overfitting
- High momentum (0.99) for stable prototype updates
- No prototype interference during training

### Balanced Sampling

- Semantic similarity-based buffer sampling
- Adaptive thresholding for semantic group formation
- Balance factors to ensure diverse representation

### GELU Activation

- Configurable GELU application across network layers
- Improved gradient flow and performance

## Scripts

### 1. `seq-cifar10.py`

Grid search script for Sequential CIFAR-10 experiments.

**Key Parameters:**

- Buffer sizes: 200, 500
- Alpha (consistency weight): 0.5 (buf200), 0.2 (buf500)
- Beta (knowledge distillation): 1.0
- Op_weight (prototype regularization): 0.5, 0.7
- Sm_weight (semantic weight): 0.01, 0.05
- Prototype momentum: 0.99
- Guidance weight: 0.05, 0.1

### 2. `seq-cifar100.py`

Direct execution script for Sequential CIFAR-100 experiments.

**Usage:**

```bash
python scripts/sarl_enhanced_gelu_balanced_inference/seq-cifar100.py --model sarl_enhanced_gelu_balanced_inference --dataset seq-cifar100 [other_args]
```

### 3. `seq-tinyimg.py`

Direct execution script for Sequential Tiny ImageNet experiments.

**Usage:**

```bash
python scripts/sarl_enhanced_gelu_balanced_inference/seq-tinyimg.py --model sarl_enhanced_gelu_balanced_inference --dataset seq-tinyimg [other_args]
```

## Model-Specific Parameters

The `sarl_enhanced_gelu_balanced_inference` model introduces several new parameters:

### Prototype Parameters

- `--prototype_momentum`: Momentum for prototype updates (default: 0.99)
- `--enable_inference_guidance`: Enable prototype guidance during inference (default: 1)
- `--guidance_weight`: Weight for prototype guidance (default: 0.05)

### Balanced Sampling Parameters

- `--use_balanced_sampling`: Enable balanced sampling (default: 1)
- `--balance_weight`: Weight for balance factor (default: 1.0)

### GELU Parameters

- `--apply_gelu`: Which layers to apply GELU to (default: [1, 1, 1, 1])
- `--num_feats`: Number of features (default: 512)

## Example Usage

### Grid Search (CIFAR-10)

```bash
cd scripts/sarl_enhanced_gelu_balanced_inference
python seq-cifar10.py
```

### Single Experiment (CIFAR-100)

```bash
python scripts/sarl_enhanced_gelu_balanced_inference/seq-cifar100.py \
    --model sarl_enhanced_gelu_balanced_inference \
    --dataset seq-cifar100 \
    --buffer_size 500 \
    --alpha 0.2 \
    --beta 1.0 \
    --op_weight 0.5 \
    --sm_weight 0.01 \
    --prototype_momentum 0.99 \
    --enable_inference_guidance 1 \
    --guidance_weight 0.05 \
    --use_balanced_sampling 1 \
    --balance_weight 1.0 \
    --apply_gelu 1 1 1 1 \
    --num_feats 512 \
    --seed 1
```

## Output

Results will be saved to:

- `experiments/sarl_enhanced_gelu_balanced_inference/`
- CSV logs with detailed metrics
- TensorBoard logs (if enabled)
- Model checkpoints (if enabled)

## Key Differences from Base SARL

1. **Inference-Only Prototypes**: No prototype guidance during training
2. **Conservative Guidance**: Very small guidance weights (0.05-0.1)
3. **High Momentum**: Stable prototype updates (0.99)
4. **Balanced Sampling**: Semantic-aware buffer management
5. **GELU Activation**: Improved activation functions

## Notes

- The inference-only approach prevents prototype interference during training
- Conservative guidance weights ensure minimal impact on model predictions
- High momentum provides stable prototype representations
- Balanced sampling improves buffer diversity and representation
