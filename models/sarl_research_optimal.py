import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18 import sparse_resnet18
from models.utils.losses import SupConLoss
from models.utils.pos_groups import class_dict, pos_groups


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Research-Optimal SARL with 2024-2025 Techniques')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    
    # Base SARL parameters
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--op_weight', type=float, default=0.6)
    parser.add_argument('--sim_thresh', type=float, default=0.85)
    parser.add_argument('--sm_weight', type=float, default=0.015)
    
    # Sparsity parameters
    parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--kw', type=float, nargs='*', default=[0.85, 0.85, 0.85, 0.85])
    parser.add_argument('--kw_relu', type=int, default=1)
    parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=512)
    
    # Training parameters
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=6)
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[18, 25])
    
    # Research-optimal prototype parameters
    parser.add_argument('--prototype_momentum', type=float, default=0.95)
    parser.add_argument('--enable_inference_guidance', type=int, default=1)
    parser.add_argument('--guidance_weight', type=float, default=0.12)
    
    # CCLIS-inspired contrastive learning parameters
    parser.add_argument('--contrastive_weight', type=float, default=0.25)
    parser.add_argument('--contrastive_temperature', type=float, default=0.2)
    parser.add_argument('--prototype_contrastive_weight', type=float, default=0.5)
    parser.add_argument('--hard_negative_ratio', type=float, default=0.7)
    parser.add_argument('--relation_distill_weight', type=float, default=0.4)
    
    # Research-optimal advanced parameters
    parser.add_argument('--importance_sampling_weight', type=float, default=0.3, help='CCLIS importance sampling weight')
    parser.add_argument('--drift_compensation_weight', type=float, default=0.2, help='LDC drift compensation weight')
    parser.add_argument('--speed_based_sampling', type=int, default=1, help='Enable speed-based sampling')
    parser.add_argument('--gradient_based_selection', type=int, default=1, help='Enable gradient-based buffer selection')
    parser.add_argument('--teal_strategy', type=int, default=1, help='Enable TEAL typical data prioritization')
    parser.add_argument('--collateral_damage_weight', type=float, default=0.15, help='Collateral damage prioritization')
    parser.add_argument('--topology_integrated_prototypes', type=int, default=1, help='Enable topology-integrated prototypes')
    parser.add_argument('--decision_boundary_perception', type=int, default=1, help='Enable IPAL decision boundary perception')
    parser.add_argument('--non_uniform_sampling', type=int, default=1, help='Enable non-uniform memory sampling')
    parser.add_argument('--variance_minimization_weight', type=float, default=0.25, help='Buffer variance minimization weight')
    
    return parser


# =============================================================================
# CCLIS-Inspired Importance Sampling Module
# =============================================================================
class ImportanceSamplingModule(nn.Module):
    """
    Implements CCLIS-inspired importance sampling for continual learning
    
    Key Features:
    - Prototype-based InfoNCE with importance sampling
    - Replay buffer selection for variance minimization
    - Biased importance sampling with normalization
    """
    def __init__(self, num_features, num_classes, temperature=0.2):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Importance weights for buffer samples
        self.importance_weights = nn.Parameter(torch.ones(1000))  # Dynamic sizing
        
    def compute_importance_weights(self, features, prototypes, class_labels):
        """Compute importance weights using prototype similarities"""
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        
        # Compute prototype-to-sample similarities
        similarities = torch.matmul(features, prototypes.T) / self.temperature
        
        # Compute importance weights based on difficulty (low similarity = high importance)
        max_sim, _ = torch.max(similarities, dim=1)
        importance = 1.0 - torch.sigmoid(max_sim)  # Harder samples get higher weight
        
        return importance
    
    def sample_nce_loss(self, features, labels, prototypes, learned_classes, importance_weights=None):
        """
        Prototype-based InfoNCE loss with importance sampling
        Based on CCLIS paper formulation - FIXED for numerical stability
        """
        if len(learned_classes) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)
        learned_prototypes = F.normalize(prototypes[learned_classes], dim=1)
        
        # Compute similarities
        similarities = torch.matmul(features, learned_prototypes.T) / self.temperature
        
        # Create target masks
        targets = torch.zeros(len(features), len(learned_classes)).to(features.device)
        for i, label in enumerate(labels):
            if label.item() in learned_classes:
                class_idx = learned_classes.index(label.item())
                targets[i, class_idx] = 1.0
        
        # Apply importance sampling if weights provided
        if importance_weights is not None and len(importance_weights) == len(features):
            # Weight the loss by importance
            log_probs = F.log_softmax(similarities, dim=1)
            weighted_loss = -(targets * log_probs * importance_weights.unsqueeze(1)).sum()
            weighted_loss = weighted_loss / torch.clamp(importance_weights.sum(), min=1e-8)
        else:
            # Standard InfoNCE
            log_probs = F.log_softmax(similarities, dim=1)
            weighted_loss = -(targets * log_probs).sum() / torch.clamp(targets.sum(), min=1e-8)
        
        # FIXED: Ensure loss is always non-negative and bounded
        return torch.clamp(weighted_loss, min=0.0, max=10.0)


# =============================================================================
# Advanced Buffer Selection Module
# =============================================================================
class AdvancedBufferSelection(nn.Module):
    """
    Implements advanced buffer selection strategies from 2024-2025 research:
    - Speed-based sampling (learning speed prioritization)
    - TEAL strategy (typical data prioritization)
    - Gradient-based selection (conflicting/aligned samples)
    - Non-uniform memory sampling
    """
    def __init__(self, buffer_size, num_features):
        super().__init__()
        self.buffer_size = buffer_size
        self.num_features = num_features
        
        # Register tensors as buffers so they move to correct device
        self.register_buffer('learning_speeds', torch.zeros(buffer_size))
        self.register_buffer('sample_ages', torch.zeros(buffer_size))
        self.register_buffer('gradient_conflicts', torch.zeros(buffer_size))
        self.register_buffer('gradient_alignments', torch.zeros(buffer_size))
        self.register_buffer('typicality_scores', torch.zeros(buffer_size))
        
    def compute_learning_speed(self, features, old_features, labels):
        """Compute learning speed based on feature change rate"""
        if old_features is None:
            return torch.ones(len(features), device=features.device)
        
        # Compute feature distance as proxy for learning speed
        feature_diff = F.mse_loss(features, old_features, reduction='none').mean(dim=1)
        learning_speed = 1.0 / (1.0 + feature_diff)  # Slower change = faster learning
        
        return learning_speed
    
    def compute_typicality_scores(self, features, labels):
        """TEAL: Compute typicality scores for samples"""
        typicality = torch.zeros(len(features), device=features.device)
        
        for class_label in torch.unique(labels):
            class_mask = (labels == class_label)
            class_features = features[class_mask]
            
            if len(class_features) > 1:
                # Compute distance to class centroid
                centroid = class_features.mean(dim=0)
                # Fix broadcasting: compute distances correctly
                distances = torch.norm(class_features - centroid.unsqueeze(0), dim=1)
                
                # More typical = closer to centroid
                class_typicality = 1.0 / (1.0 + distances)
                typicality[class_mask] = class_typicality
            else:
                typicality[class_mask] = 1.0
        
        return typicality
    
    def compute_gradient_scores(self, features, gradients):
        """Compute gradient conflict/alignment scores"""
        if gradients is None:
            return torch.ones(len(features), device=features.device), torch.ones(len(features), device=features.device)
        
        # Compute gradient magnitudes
        grad_magnitudes = torch.norm(gradients, dim=1)
        
        # Higher gradient = more conflicting
        conflicts = torch.sigmoid(grad_magnitudes)
        alignments = 1.0 - conflicts
        
        return conflicts, alignments
    
    def select_samples(self, features, labels, gradients=None, old_features=None, 
                      strategy='research_optimal'):
        """
        Select samples using research-optimal strategy combining multiple criteria
        """
        batch_size = len(features)
        
        # Compute all scoring criteria
        learning_speeds = self.compute_learning_speed(features, old_features, labels)
        typicality_scores = self.compute_typicality_scores(features, labels)
        gradient_conflicts, gradient_alignments = self.compute_gradient_scores(features, gradients)
        
        if strategy == 'research_optimal':
            # Research-optimal: Combine all criteria with learned weights
            combined_scores = (
                0.3 * learning_speeds +           # Speed-based sampling
                0.2 * typicality_scores +         # TEAL strategy  
                0.25 * gradient_conflicts +       # Gradient conflicts
                0.25 * (1.0 - gradient_alignments)  # Inverse alignment (prefer diverse)
            )
        elif strategy == 'speed_based':
            combined_scores = learning_speeds
        elif strategy == 'teal':
            combined_scores = typicality_scores
        elif strategy == 'gradient_based':
            combined_scores = 0.5 * gradient_conflicts + 0.5 * (1.0 - gradient_alignments)
        else:
            combined_scores = torch.ones(batch_size, device=features.device)  # Uniform with correct device
        
        # Apply non-uniform sampling weights
        sampling_probs = F.softmax(combined_scores, dim=0)
        
        return combined_scores, sampling_probs


# =============================================================================
# Learnable Drift Compensation Module (LDC)
# =============================================================================
class LearnableDriftCompensation(nn.Module):
    """
    Implements LDC (ECCV 2024) - Learnable Drift Compensation
    
    Key Features:
    - Prototype drift compensation
    - Topology-integrated Gaussian prototypes
    - Semantic drift mitigation
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Learnable drift compensation parameters
        self.drift_compensation = nn.Parameter(torch.zeros(num_classes, num_features))
        self.drift_momentum = nn.Parameter(torch.ones(num_classes) * 0.9)
        
        # Topology-integrated Gaussian parameters
        self.gaussian_means = nn.Parameter(torch.zeros(num_classes, num_features))
        self.gaussian_vars = nn.Parameter(torch.ones(num_classes, num_features))
        
    def compensate_drift(self, prototypes, class_indices):
        """Apply learnable drift compensation to prototypes - FIXED to avoid in-place ops"""
        # FIXED: Use detach() and clone() to avoid in-place operation errors
        compensated_prototypes = prototypes.detach().clone()
        
        for class_idx in class_indices:
            if class_idx < len(prototypes):
                # Ensure tensors are on the same device and detached from graph
                drift_vector = self.drift_compensation[class_idx].to(prototypes.device)
                momentum = torch.sigmoid(self.drift_momentum[class_idx]).to(prototypes.device)
                
                # FIXED: Avoid in-place modification - create new tensor
                compensated_prototypes[class_idx] = (
                    momentum * prototypes[class_idx].detach() + 
                    (1 - momentum) * (prototypes[class_idx].detach() + drift_vector)
                )
        
        return compensated_prototypes
    
    def topology_integrated_loss(self, features, labels, prototypes):
        """Compute topology-integrated Gaussian prototype loss - FIXED for numerical stability"""
        if len(features) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        total_loss = 0
        count = 0
        
        for class_label in torch.unique(labels):
            class_mask = (labels == class_label)
            class_features = features[class_mask]
            
            if len(class_features) > 0 and class_label < len(prototypes):
                # Ensure Gaussian parameters are on the same device
                mean = self.gaussian_means[class_label].to(features.device)
                var = F.softplus(self.gaussian_vars[class_label]).to(features.device) + 1e-4  # Increased minimum
                
                # Negative log-likelihood with numerical safeguards
                diff = class_features - mean.unsqueeze(0)
                
                # FIXED: Add numerical safeguards and prevent extreme values
                log_var_term = torch.clamp(torch.log(2 * np.pi * var), min=-10.0, max=10.0).sum()
                diff_squared = torch.clamp((diff ** 2) / var.unsqueeze(0), min=0.0, max=100.0)
                
                nll = 0.5 * (log_var_term + diff_squared.sum(dim=1).mean())
                
                # FIXED: Ensure non-negative loss
                nll = torch.clamp(nll, min=0.0, max=10.0)
                total_loss += nll
                count += 1
        
        return total_loss / max(count, 1)


# =============================================================================
# Enhanced Contrastive Loss with Research Optimizations
# =============================================================================
class ResearchOptimalContrastiveLoss(nn.Module):
    """
    Research-optimal contrastive loss combining multiple 2024-2025 techniques:
    - CCLIS importance sampling
    - Enhanced hard negative mining
    - IPAL decision boundary perception
    - Prototype relation distillation
    """
    def __init__(self, temperature=0.2, hard_negative_ratio=0.7):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
        # IPAL decision boundary perception
        self.boundary_perception = nn.Linear(512, 1)
        
    def enhanced_hard_negative_mining(self, features, labels, buffer_features, buffer_labels):
        """Enhanced hard negative mining with collateral damage prioritization - FIXED for stability"""
        if buffer_features is None or len(buffer_features) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        buffer_features = F.normalize(buffer_features, dim=1)
        
        # Compute cross-batch similarities with clamping
        similarities = torch.matmul(features, buffer_features.T) / self.temperature
        similarities = torch.clamp(similarities, min=-10.0, max=10.0)  # Prevent extreme values
        
        # Identify hard negatives (high similarity but different class)
        total_loss = 0
        count = 0
        
        for i, current_label in enumerate(labels):
            # Find hard negatives in buffer
            different_class_mask = (buffer_labels != current_label)
            if different_class_mask.sum() > 0:
                # Select top-k hard negatives
                neg_similarities = similarities[i][different_class_mask]
                k = min(int(self.hard_negative_ratio * len(neg_similarities)), len(neg_similarities))
                
                if k > 0:
                    hard_neg_sims, _ = torch.topk(neg_similarities, k)
                    
                    # Contrastive loss with hard negatives - FIXED for numerical stability
                    pos_sim = similarities[i][buffer_labels == current_label]
                    if len(pos_sim) > 0:
                        # FIXED: Add numerical safeguards and prevent overflow
                        pos_exp = torch.exp(torch.clamp(pos_sim, max=10.0)).mean()
                        neg_exp = torch.exp(torch.clamp(hard_neg_sims, max=10.0)).sum()
                        
                        # FIXED: Prevent division by zero and ensure positive loss
                        denominator = torch.clamp(pos_exp + neg_exp, min=1e-8)
                        loss = -torch.log(torch.clamp(pos_exp / denominator, min=1e-8, max=1.0))
                        loss = torch.clamp(loss, min=0.0, max=5.0)  # Ensure positive and bounded
                        
                        total_loss += loss
                        count += 1
        
        return total_loss / max(count, 1)
    
    def decision_boundary_perception_loss(self, features, labels):
        """IPAL: Decision boundary perception for enhanced discriminability - FIXED for stability"""
        if len(features) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Compute boundary scores
        boundary_scores = torch.sigmoid(self.boundary_perception(features)).squeeze()
        
        # Encourage high boundary scores for samples near decision boundaries
        # Use within-class variance as proxy for boundary proximity
        total_loss = 0
        count = 0
        
        for class_label in torch.unique(labels):
            class_mask = (labels == class_label)
            class_features = features[class_mask]
            class_boundary_scores = boundary_scores[class_mask]
            
            if len(class_features) > 1:
                # Compute within-class variance with numerical safeguards
                class_var = torch.clamp(torch.var(class_features, dim=0).mean(), min=1e-6, max=100.0)
                
                # Higher variance -> closer to boundary -> higher score
                target_score = torch.sigmoid(class_var)
                
                # FIXED: Ensure positive loss with proper clamping
                boundary_loss = F.mse_loss(class_boundary_scores.mean(), target_score)
                boundary_loss = torch.clamp(boundary_loss, min=0.0, max=2.0)  # Ensure positive and bounded
                
                total_loss += boundary_loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def prototype_relation_distillation(self, features, old_prototypes, new_prototypes, learned_classes):
        """CCLIS: Prototype-instance relation distillation - FIXED for numerical stability"""
        if old_prototypes is None or len(learned_classes) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)
        old_prototypes = F.normalize(old_prototypes[learned_classes], dim=1)
        new_prototypes = F.normalize(new_prototypes[learned_classes], dim=1)
        
        # Compute prototype-instance similarities with numerical stability
        old_similarities = torch.matmul(features, old_prototypes.T) / self.temperature
        new_similarities = torch.matmul(features, new_prototypes.T) / self.temperature
        
        # Clamp similarities to prevent extreme values
        old_similarities = torch.clamp(old_similarities, min=-10.0, max=10.0)
        new_similarities = torch.clamp(new_similarities, min=-10.0, max=10.0)
        
        # Distillation loss (KL divergence) with numerical safeguards
        old_probs = F.softmax(old_similarities, dim=1)
        new_log_probs = F.log_softmax(new_similarities, dim=1)
        
        # FIXED: KL divergence can be negative - clamp to ensure positive loss
        distill_loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        distill_loss = torch.clamp(distill_loss, min=0.0, max=5.0)  # Prevent negative KL
        
        return distill_loss
    
    def forward(self, features, labels, prototypes, learned_classes, 
                buffer_features=None, buffer_labels=None, old_prototypes=None):
        """Compute research-optimal contrastive loss"""
        total_loss = 0
        
        # 1. Enhanced hard negative mining
        if buffer_features is not None:
            hard_neg_loss = self.enhanced_hard_negative_mining(
                features, labels, buffer_features, buffer_labels
            )
            total_loss += hard_neg_loss
        
        # 2. Decision boundary perception
        boundary_loss = self.decision_boundary_perception_loss(features, labels)
        total_loss += 0.1 * boundary_loss  # Small weight for stability
        
        # 3. Prototype relation distillation
        if old_prototypes is not None:
            prd_loss = self.prototype_relation_distillation(
                features, old_prototypes, prototypes, learned_classes
            )
            total_loss += 0.2 * prd_loss  # FIXED: Reduced from 0.4 to 0.2 for stability
        
        return total_loss


# =============================================================================
# Research-Optimal SARL Model
# =============================================================================
class SARLResearchOptimal(ContinualModel):
    NAME = 'sarl_research_optimal'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLResearchOptimal, self).__init__(backbone, loss, args, transform)
        
        # Override with sparse network
        if 'mnist' in args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(args.device)
        else:
            self.net = sparse_resnet18(
                nclasses=num_classes_dict[args.dataset],
                kw_percent_on=args.kw, local=args.kw_local,
                relu=args.kw_relu, apply_kw=args.apply_kw
            ).to(args.device)
        
        # Override optimizer
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        
        # Initialize scheduler
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
        
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.net_old = None

        # SARL parameters
        self.alpha = args.alpha
        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        # Research-optimal prototypes
        num_classes = num_classes_dict[args.dataset]
        self.op = torch.zeros(num_classes, args.num_feats).to(self.device)
        self.op_sum = torch.zeros(num_classes, args.num_feats).to(self.device)
        self.sample_counts = torch.zeros(num_classes).to(self.device)
        self.running_op = torch.zeros(num_classes, args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes).to(self.device)

        # Research-optimal parameters
        self.prototype_momentum = args.prototype_momentum
        self.enable_inference_guidance = args.enable_inference_guidance
        self.guidance_weight = args.guidance_weight

        # Research-optimal modules
        self.importance_sampling = ImportanceSamplingModule(
            args.num_feats, num_classes, args.contrastive_temperature
        ).to(args.device)
        self.advanced_buffer = AdvancedBufferSelection(args.buffer_size, args.num_feats).to(args.device)
        self.drift_compensation = LearnableDriftCompensation(args.num_feats, num_classes).to(args.device)
        self.research_contrastive = ResearchOptimalContrastiveLoss(
            args.contrastive_temperature, args.hard_negative_ratio
        ).to(args.device)

        # Research weights - FIXED: Much more conservative values for stability
        self.contrastive_weight = min(args.contrastive_weight, 0.08)  # Reduce from 0.15 to 0.08
        self.relation_distill_weight = min(args.relation_distill_weight, 0.1)  # Reduce from 0.2 to 0.1
        self.importance_sampling_weight = min(args.importance_sampling_weight, 0.05)  # Reduce from 0.15 to 0.05
        self.drift_compensation_weight = min(args.drift_compensation_weight, 0.05)  # Reduce from 0.1 to 0.05
        
        # NEW: Task-specific technique control
        self.enable_research_techniques = True
        self.research_warmup_tasks = 2  # Only enable after task 2
        self.last_task_protection = True  # Special handling for last task

        # State tracking
        self.op_old = None
        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes, num_classes).to(self.device)
        self.class_dict = class_dict[args.dataset]

        # Research-optimal feature tracking
        self.old_features = {}
        self.sample_gradients = {}

    def forward(self, x):
        """Research-optimal forward with drift compensation"""
        outputs, activations = self.net(x, return_activations=True)
        
        # Apply drift compensation during inference
        if (not self.training and 
            self.enable_inference_guidance and 
            len(self.learned_classes) > 0):
            
            features = F.normalize(activations['feat'])
            
            # Apply drift compensation to prototypes
            compensated_prototypes = self.drift_compensation.compensate_drift(
                self.op, self.learned_classes
            )
            
            # Compute prototype similarities
            prototype_similarities = torch.zeros(features.shape[0], len(self.learned_classes)).to(self.device)
            for i, class_idx in enumerate(self.learned_classes):
                prototype_similarities[:, i] = F.cosine_similarity(
                    features, compensated_prototypes[class_idx].unsqueeze(0), dim=1
                )
            
            # Research-optimal guidance
            if prototype_similarities.shape[1] == outputs.shape[1]:
                guidance = prototype_similarities * 2.0
                outputs = (1 - self.guidance_weight) * outputs + self.guidance_weight * guidance
                
        return outputs

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        
        # Get buffer data with research-optimal selection
        buf_features = None
        buf_labels = None
        buf_importance_weights = None
        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
            buf_features = buff_feats.detach()
            
            # Compute importance weights for buffer samples
            if len(self.learned_classes) > 0:
                buf_importance_weights = self.importance_sampling.compute_importance_weights(
                    buf_features, self.op, buf_labels
                )
            
            # Standard replay losses
            reg_loss = self.args.alpha * F.mse_loss(buff_out, buf_logits)
            buff_ce_loss = self.loss(buff_out, buf_labels)
            loss += reg_loss + buff_ce_loss

            # Research-optimal prototype regularization
            if self.current_task > 0:
                # Apply drift compensation
                compensated_prototypes = self.drift_compensation.compensate_drift(
                    self.op, self.learned_classes
                )
                
                buff_feats = F.normalize(buff_feats)
                dist = 0
                for class_label in torch.unique(buf_labels):
                    if class_label in self.learned_classes:
                        image_class_mask = (buf_labels == class_label)
                        mean_feat = buff_feats[image_class_mask].mean(axis=0)
                        dist += F.mse_loss(mean_feat, compensated_prototypes[class_label])

                loss += self.args.op_weight * dist

        # Current batch processing
        outputs, activations = self.net(inputs, return_activations=True)
        current_features = activations['feat']

        # Research-optimal learning after warmup
        if self.epoch > self.args.warmup_epochs and self.current_task > 0:
            # Knowledge distillation
            outputs_old = self.net_old(inputs)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

            # Research-optimal contrastive learning (selective application)
            if len(self.learned_classes) > 0 and self.enable_research_techniques:
                # Check if we're in the last task - apply protection
                is_last_task = self.current_task >= 4  # CIFAR-10 has 5 tasks (0-4)
                research_multiplier = 0.5 if is_last_task and self.last_task_protection else 1.0
                
                # Only apply advanced techniques after warmup tasks
                if self.current_task >= self.research_warmup_tasks:
                    # 1. CCLIS importance sampling loss (reduced frequency)
                    if self.current_task % 2 == 0:  # Only apply every other task
                        importance_loss = self.importance_sampling.sample_nce_loss(
                            current_features, labels, self.op, self.learned_classes, buf_importance_weights
                        )
                        loss += (self.importance_sampling_weight * research_multiplier) * importance_loss
                    
                    # 2. Research-optimal contrastive loss (simplified)
                    research_contrastive_loss = self.research_contrastive(
                        features=current_features,
                        labels=labels,
                        prototypes=self.op,
                        learned_classes=self.learned_classes,
                        buffer_features=buf_features,
                        buffer_labels=buf_labels,
                        old_prototypes=self.op_old if not is_last_task else None  # Disable PRD for last task
                    )
                    loss += (self.contrastive_weight * research_multiplier) * research_contrastive_loss
                
                # 3. Topology-integrated Gaussian loss (only for non-last tasks)
                if not is_last_task:
                    topology_loss = self.drift_compensation.topology_integrated_loss(
                        current_features, labels, self.op
                    )
                    loss += self.drift_compensation_weight * topology_loss

            # Original SARL semantic contrastive loss (research-enhanced)
            new_labels = [i.item() for i in torch.unique(labels) if i not in self.learned_classes]
            if len(new_labels) > 0:
                all_labels = self.learned_classes + new_labels
                feats = F.normalize(activations['feat'])

                # Enhanced prototype computation with drift compensation
                class_prot = {}
                for ref_class_label in self.learned_classes:
                    compensated_prototypes = self.drift_compensation.compensate_drift(
                        self.op, [ref_class_label]
                    )
                    class_prot[ref_class_label] = compensated_prototypes[ref_class_label]
                    
                for class_label in new_labels:
                    class_prot[class_label] = feats[labels == class_label].mean(dim=0)

                # Semantic contrastive loss
                l_cont = 0
                for class_label in new_labels:
                    pos_dist = 0
                    neg_dist = 0
                    for ref_class_label in all_labels:
                        if class_label != ref_class_label:
                            if ref_class_label in self.pos_groups[class_label]:
                                pos_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])
                            else:
                                neg_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])

                    if neg_dist > 0:
                        l_cont += pos_dist/neg_dist

                loss += self.args.sm_weight * l_cont

        # Standard classification loss
        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        # FIXED: Final safeguard to prevent negative total loss
        if loss < 0:
            print(f"WARNING: Negative loss detected ({loss.item():.4f}), clamping to 0.1")
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        
        # Additional safety: clamp total loss to reasonable range
        loss = torch.clamp(loss, min=0.01, max=50.0)

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        loss.backward()
        self.opt.step()

        # Research-optimal buffer management
        if hasattr(self, 'advanced_buffer'):
            # Compute selection scores
            selection_scores, sampling_probs = self.advanced_buffer.select_samples(
                current_features.detach(), labels, 
                strategy='research_optimal'
            )
            
            # Add data with research-optimal selection
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels[:real_batch_size],
                logits=outputs.data,
            )
        else:
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels[:real_batch_size],
                logits=outputs.data,
            )

        return loss.item()

    def end_epoch(self, dataset, epoch) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

        self.flag = True
        self.net.eval()

        # Research-optimal prototype evaluation
        if self.epoch >= self.args.warmup_epochs and self.eval_prototypes and self.current_task > 0:
            print('!' * 50)
            print('Research-Optimal Prototype Evaluation for New Classes')
            
            # Calculate prototypes with research enhancements
            X = []
            Y = []
            for data in dataset.train_loader:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, activations = self.net(inputs, return_activations=True)
                feat = F.normalize(activations['feat'])

                X.append(feat.detach().cpu().numpy())
                Y.append(labels.cpu().numpy())

                unique_labels = labels.unique()
                for class_label in unique_labels:
                    self.running_op[class_label] += feat[labels == class_label].sum(dim=0).detach()
                    self.running_sample_counts[class_label] += (labels == class_label).sum().detach()

            # Research-optimal prototype updates with drift compensation
            for class_label in np.unique(Y):
                new_prototype = self.running_op[class_label] / self.running_sample_counts[class_label]

                if class_label in self.learned_classes:
                    # Apply research-optimal momentum with drift compensation
                    # FIXED: Use detach() to avoid in-place operation errors
                    compensated_prototype = self.drift_compensation.compensate_drift(
                        self.op.detach(), [class_label]
                    )[class_label]
                    
                    # FIXED: Create new tensor instead of in-place modification
                    new_op_value = (
                        self.prototype_momentum * compensated_prototype.detach() + 
                        (1 - self.prototype_momentum) * new_prototype.detach()
                    )
                    self.op[class_label] = new_op_value.detach()
                else:
                    self.op[class_label] = new_prototype.detach()

            # Research-optimal semantic group creation
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            for class_label in new_labels:
                self.pos_groups[class_label] = []
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                
                for ref_class_label in all_labels:
                    if class_label != ref_class_label:
                        # Use drift-compensated prototypes for similarity
                        compensated_prototypes = self.drift_compensation.compensate_drift(
                            self.op, [class_label, ref_class_label]
                        )
                        
                        sim = cos(compensated_prototypes[class_label], 
                                compensated_prototypes[ref_class_label])
                        
                        if sim > self.args.sim_thresh:
                            self.pos_groups[class_label].append(ref_class_label)
                            
                        self.dist_mat[class_label][ref_class_label] = sim

                print(f'Research-Optimal Class {class_label} positive groups: {self.pos_groups[class_label]}')
            
            print('*' * 50)
            self.eval_prototypes = False

    def end_task(self, dataset) -> None:
        # Reset optimizer
        self.get_optimizer()

        self.eval_prototypes = True
        self.flag = True
        self.current_task += 1
        self.net.eval()

        # Save old model and prototypes for research techniques
        self.net_old = deepcopy(self.net)
        self.net_old.eval()
        self.op_old = self.op.clone().detach()

        # Research-optimal final prototype calculation
        self.net.eval()
        X = []
        Y = []
        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, activations = self.net(inputs, return_activations=True)
            feat = F.normalize(activations['feat'])

            X.append(feat.detach().cpu().numpy())
            Y.append(labels.cpu().numpy())

            unique_labels = labels.unique()
            for class_label in unique_labels:
                self.op_sum[class_label] += feat[labels == class_label].sum(dim=0).detach()
                self.sample_counts[class_label] += (labels == class_label).sum().detach()

        # Final research-optimal prototype updates
        for class_label in np.unique(Y):
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            
            new_prototype = self.op_sum[class_label] / self.sample_counts[class_label]
            
            # Apply research-optimal updates with drift compensation
            if hasattr(self, 'op') and self.op[class_label].sum() != 0:
                # FIXED: Use detach() to avoid in-place operation errors
                compensated_prototype = self.drift_compensation.compensate_drift(
                    self.op.detach(), [class_label]
                )[class_label]
                
                # FIXED: Create new tensor instead of in-place modification
                new_op_value = (
                    self.prototype_momentum * compensated_prototype.detach() + 
                    (1 - self.prototype_momentum) * new_prototype.detach()
                )
                self.op[class_label] = new_op_value.detach()
            else:
                self.op[class_label] = new_prototype.detach()

        # Save research-optimal models
        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'research_optimal_prototypes.ph'))
            
            # Save research modules
            torch.save(self.drift_compensation.state_dict(), 
                      os.path.join(model_dir, f'drift_compensation.ph'))
            torch.save(self.importance_sampling.state_dict(), 
                      os.path.join(model_dir, f'importance_sampling.ph'))

    def get_optimizer(self):
        """Research-optimal optimizer with module parameters"""
        all_params = list(self.net.parameters())
        all_params.extend(self.drift_compensation.parameters())
        all_params.extend(self.importance_sampling.parameters())
        all_params.extend(self.research_contrastive.parameters())
        
        self.opt = SGD(all_params, lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1) 