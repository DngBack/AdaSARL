# Enhanced SARL: Contrastive-Based Semantic Aware Representation Learning

## Overview

This is an enhanced version of the original SARL (Semantic Aware Representation Learning) that replaces cosine similarity with learned contrastive-based semantic similarity for more principled and adaptive semantic grouping.

## Key Improvements

### 1. **Learnable Semantic Similarity Network**

- **Original**: Fixed cosine similarity with arbitrary threshold
- **Enhanced**: Neural network that learns to compute semantic similarity
- **Benefits**: Adapts to dataset characteristics, learns nuanced relationships

```python
class SemanticSimilarity(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        self.similarity_net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

### 2. **Soft Semantic Weighting**

- **Original**: Binary grouping (similar/dissimilar)
- **Enhanced**: Continuous semantic weights (0-1)
- **Benefits**: More flexible relationships, gradual semantic learning

```python
# Compute soft semantic weights for all class pairs
semantic_weights = self.compute_semantic_weights(new_labels, all_labels)
```

### 3. **Enhanced Contrastive Loss**

- **Original**: Ratio-based loss with binary grouping
- **Enhanced**: Weighted contrastive loss with learned similarity
- **Benefits**: Direct optimization of semantic relationships

```python
def semantic_contrastive_loss(self, prototypes, semantic_weights):
    # Weighted contrastive loss with semantic guidance
    weighted_pos_loss = (sem_weights * distances).sum()
    weighted_neg_loss = ((1 - sem_weights) * distances).sum()
    return weighted_pos_loss / weighted_neg_loss
```

### 4. **End-to-End Semantic Learning**

- **Original**: Static similarity computation
- **Enhanced**: Similarity network trained with positive/negative pairs
- **Benefits**: Semantic relationships evolve during training

```python
def update_semantic_similarity(self):
    # Create positive/negative pairs and train similarity network
    pos_pairs = self.create_positive_pairs()
    neg_pairs = self.create_negative_pairs()
    # Train with binary cross-entropy loss
```

## Architecture Changes

### New Components Added

1. **SemanticSimilarity Network**: Learns to compute similarity between class prototypes
2. **AdaptiveThreshold**: Learnable threshold for semantic grouping (future enhancement)
3. **Semantic Weight Computation**: Soft weighting instead of binary grouping
4. **Enhanced Loss Functions**: Semantic-aware contrastive loss

### Modified Components

1. **Prototype Computation**: Same as original but used with learned similarity
2. **Training Loop**: Added semantic network training
3. **Loss Computation**: Replaced ratio-based loss with weighted contrastive loss

## Usage

### Running Enhanced SARL

```bash
# Run enhanced SARL on CIFAR-100
python scripts/sarl_enhanced/seq-cifar100.py

# Or run individual experiment
python main.py \
    --model sarl_enhanced \
    --dataset seq-cifar100 \
    --buffer_size 200 \
    --sim_lr 0.001 \
    --sm_weight 0.01 \
    --experiment_id enhanced_test
```

### Comparison with Original

```bash
# Run comparison between original and enhanced SARL
python compare_sarl_models.py
```

## Hyperparameters

### New Parameters

- `--sim_lr`: Learning rate for semantic similarity network (default: 0.001)
- `--sm_weight`: Weight for semantic contrastive loss (default: 0.01)

### Modified Parameters

- Removed `--sim_thresh`: No longer needed (replaced by learned similarity)
- Enhanced `--sm_weight`: Now controls weighted contrastive loss

## Expected Benefits

### 1. **Better Semantic Representations**

- Learned similarity captures more nuanced relationships
- Adapts to dataset-specific semantic structures
- More robust to domain shifts

### 2. **Improved Continual Learning**

- Semantic relationships evolve during training
- Better adaptation to new classes
- Reduced catastrophic forgetting

### 3. **More Flexible Grouping**

- Soft weights allow gradual semantic learning
- No arbitrary threshold selection
- Better handling of edge cases

### 4. **End-to-End Optimization**

- Semantic similarity directly optimized for the task
- No manual parameter tuning required
- More principled approach

## Implementation Details

### Semantic Network Training

The semantic similarity network is trained using positive and negative pairs:

**Positive Pairs**: Classes that should be semantically similar

- Animals: cat, dog, horse, etc.
- Vehicles: car, truck, bus, etc.
- Plants: tree, flower, grass, etc.

**Negative Pairs**: Classes that should be semantically different

- Cross-category pairs: cat vs. car, tree vs. bus, etc.

### Loss Function

The enhanced contrastive loss uses learned semantic weights:

```python
L_semantic = Σ (w_ij * d_ij) / Σ ((1 - w_ij) * d_ij)
```

Where:

- `w_ij`: Learned semantic weight between classes i and j
- `d_ij`: Distance between prototypes of classes i and j

### Training Schedule

1. **Warmup Phase**: Train basic representations (no semantic learning)
2. **Semantic Learning Phase**: Start training similarity network
3. **End-of-Task**: Consolidate prototypes and semantic relationships

## Results and Analysis

The enhanced model should show:

1. **Better Semantic Grouping**: More meaningful class clusters
2. **Improved Accuracy**: Better performance on continual learning tasks
3. **Adaptive Behavior**: Semantic relationships that evolve with data
4. **Robustness**: Better handling of diverse datasets

## Future Enhancements

1. **Multi-Scale Semantic Learning**: Consider relationships at different scales
2. **Hierarchical Semantic Structure**: Learn hierarchical class relationships
3. **Dynamic Thresholding**: Adaptive threshold based on data distribution
4. **Cross-Dataset Transfer**: Transfer learned semantic relationships

## Files Structure

```
models/
├── sarl.py              # Original SARL implementation
├── sarl_enhanced.py     # Enhanced SARL with contrastive learning
└── utils/
    ├── losses.py        # Loss functions
    └── pos_groups.py    # Semantic group definitions

scripts/
├── sarl/               # Original SARL scripts
└── sarl_enhanced/      # Enhanced SARL scripts

compare_sarl_models.py  # Comparison script
```

## Citation

If you use this enhanced implementation, please cite both the original SARL paper and mention the enhancements:

```bibtex
@inproceedings{sarfrazsemantic,
  title={Semantic Aware Representation Learning for Lifelong Learning},
  author={Sarfraz, Fahad and Arani, Elahe and Zonooz, Bahram},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Contact

For questions about the enhanced implementation, please refer to the original SARL repository and mention the contrastive-based enhancements.
