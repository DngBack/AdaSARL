import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.GeluMNISTMLP import GeluMNISTMLP
from backbone.GeluResNet18 import gelu_resnet18
from models.utils.losses import SupConLoss
from models.utils.pos_groups import class_dict


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
}


class SemanticSimilarity(nn.Module):
    """Learnable semantic similarity network"""
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
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
    
    def forward(self, feat1, feat2):
        # Concatenate features and compute similarity
        combined = torch.cat([feat1, feat2], dim=-1)
        return self.similarity_net(combined).squeeze(-1)


class AdaptiveThreshold(nn.Module):
    """Learnable threshold for semantic grouping"""
    def __init__(self, init_threshold=0.8):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
    
    def forward(self, similarity_scores):
        return similarity_scores > self.threshold


class BalancedBuffer(Buffer):
    """Enhanced buffer with balanced instance sampling based on semantic groups"""
    
    def __init__(self, buffer_size, device, semantic_similarity, adaptive_threshold, num_feats):
        super().__init__(buffer_size, device)
        self.semantic_similarity = semantic_similarity
        self.adaptive_threshold = adaptive_threshold
        self.num_feats = num_feats
        self.semantic_groups = {}  # Track semantic group assignments
        self.group_counts = {}     # Track group sizes
        self.sample_features = {}  # Store features for semantic computation
        self.epsilon = 1e-6        # Small constant to prevent division by zero
        
    def add_data(self, examples, labels, logits, features=None):
        """Add data to buffer with semantic group tracking"""
        super().add_data(examples, labels, logits)
        
        # Store features for semantic computation
        if features is not None:
            for i, (example, label, logit) in enumerate(zip(examples, labels, logits)):
                idx = len(self.examples) - len(examples) + i
                self.sample_features[idx] = features[i].detach()
                
                # Assign semantic group based on learned similarity
                if hasattr(self, 'op') and label.item() in self.op:
                    prototype = self.op[label.item()]
                    sim_score = self.semantic_similarity(features[i], prototype).item()
                    group_id = self.adaptive_threshold(torch.tensor([sim_score])).item()
                    
                    if group_id not in self.semantic_groups:
                        self.semantic_groups[group_id] = []
                        self.group_counts[group_id] = 0
                    
                    self.semantic_groups[group_id].append(idx)
                    self.group_counts[group_id] += 1
    
    def get_balanced_data(self, minibatch_size, transform=None, op_prototypes=None):
        """Get balanced data based on semantic group distribution"""
        if self.is_empty():
            return None, None, None
        
        # Compute semantic group balance scores
        total_groups = len(self.group_counts) if self.group_counts else 1
        balance_factors = {}
        
        for group_id, count in self.group_counts.items():
            if count > 0:
                balance_factors[group_id] = total_groups / count
            else:
                balance_factors[group_id] = 1.0
        
        # Calculate importance scores for each sample
        importance_scores = []
        valid_indices = []
        
        for idx in range(len(self.examples)):
            if idx in self.sample_features and idx in self.labels:
                label = self.labels[idx]
                feature = self.sample_features[idx]
                
                # Find semantic group for this sample
                group_id = 0  # Default group
                for gid, indices in self.semantic_groups.items():
                    if idx in indices:
                        group_id = gid
                        break
                
                # Compute semantic similarity to prototype
                if op_prototypes is not None and label.item() in op_prototypes:
                    prototype = op_prototypes[label.item()]
                    sim_score = self.semantic_similarity(feature, prototype).item()
                else:
                    sim_score = 0.5  # Default similarity
                
                # Compute importance score with balance factor
                balance_factor = balance_factors.get(group_id, 1.0)
                importance = (1.0 / (sim_score + self.epsilon)) * balance_factor
                
                importance_scores.append(importance)
                valid_indices.append(idx)
        
        if not valid_indices:
            # Fallback to uniform sampling
            return super().get_data(minibatch_size, transform)
        
        # Convert to tensor and normalize
        importance_scores = torch.tensor(importance_scores, device=self.device)
        importance_scores = importance_scores / importance_scores.sum()
        
        # Sample based on importance scores
        num_samples = min(minibatch_size, len(valid_indices))
        sampled_indices = torch.multinomial(importance_scores, num_samples, replacement=False)
        
        # Get sampled data
        buf_examples = []
        buf_labels = []
        buf_logits = []
        
        for idx in sampled_indices:
            sample_idx = valid_indices[idx]
            buf_examples.append(self.examples[sample_idx])
            buf_labels.append(self.labels[sample_idx])
            buf_logits.append(self.logits[sample_idx])
        
        buf_examples = torch.stack(buf_examples)
        buf_labels = torch.stack(buf_labels)
        buf_logits = torch.stack(buf_logits)
        
        if transform is not None:
            buf_examples = transform(buf_examples)
        
        return buf_examples, buf_labels, buf_logits


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Enhanced SARL with GELU, Balanced Sampling, and Inference-Only Prototypes')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    parser.add_argument('--sim_lr', type=float, default=0.001)
    # GELU param
    parser.add_argument('--apply_gelu', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--num_feats', type=int, default=512)
    # Balanced Sampling params
    parser.add_argument('--use_balanced_sampling', type=int, default=1)
    parser.add_argument('--balance_weight', type=float, default=1.0)
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[70, 90])
    
    # FIXED: Prototype Parameters (much more conservative)
    parser.add_argument('--prototype_momentum', type=float, default=0.99, help='High momentum for stable updates')
    parser.add_argument('--enable_inference_guidance', type=float, default=1, help='Enable prototype guidance during inference')
    parser.add_argument('--guidance_weight', type=float, default=0.05, help='Weight for prototype guidance (very small)')
    
    return parser


class ADASARLInference(ContinualModel):
    NAME = 'adasarl_inference'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ADASARLInference, self).__init__(backbone, loss, args, transform)
        
        # Initialize semantic components first
        self.semantic_similarity = SemanticSimilarity(args.num_feats).to(self.device)
        self.adaptive_threshold = AdaptiveThreshold().to(self.device)
        
        # Initialize balanced buffer
        if args.use_balanced_sampling:
            self.buffer = BalancedBuffer(
                self.args.buffer_size, 
                self.device, 
                self.semantic_similarity, 
                self.adaptive_threshold, 
                args.num_feats
            )
        else:
            self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        if 'mnist' in self.args.dataset:
            self.net = GeluMNISTMLP(28 * 28, 10).to(self.device)
        else:
            self.net = gelu_resnet18(
                nclasses=num_classes_dict[args.dataset],
                apply_gelu=args.apply_gelu
            ).to(self.device)

        self.net_old = None
        self.get_optimizer()

        # set regularization weight
        self.alpha = args.alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        # FIXED: Simple single prototypes with high momentum (no multi-centroid complexity)
        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.op_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        # FIXED: Conservative parameters
        self.prototype_momentum = args.prototype_momentum  # 0.99 for stability
        self.enable_inference_guidance = args.enable_inference_guidance
        self.guidance_weight = args.guidance_weight  # 0.05 very small

        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        
        # Enhanced semantic learning components
        self.semantic_weights = {}
        self.sim_optimizer = Adam(self.semantic_similarity.parameters(), lr=args.sim_lr)
        self.threshold_optimizer = Adam(self.adaptive_threshold.parameters(), lr=args.sim_lr)
        
        self.class_dict = class_dict[args.dataset]

    def forward(self, x):
        """FIXED: Only use prototypes during EVALUATION and with minimal weight"""
        outputs, activations = self.net(x, return_activations=True)
        
        # CRITICAL FIX: Only during inference (eval mode) and with very small weight
        if (not self.training and 
            self.enable_inference_guidance and 
            hasattr(self, 'op') and 
            self.op.sum() != 0 and 
            len(self.learned_classes) > 0):
            
            features = F.normalize(activations['feat'])
            
            # Simple cosine similarity to learned prototypes
            prototype_similarities = torch.zeros(features.shape[0], len(self.learned_classes)).to(self.device)
            for i, class_idx in enumerate(self.learned_classes):
                prototype_similarities[:, i] = F.cosine_similarity(
                    features, self.op[class_idx].unsqueeze(0), dim=1
                )
            
            # Very conservative guidance (5% influence maximum)
            if prototype_similarities.shape[1] == outputs.shape[1]:
                guidance = prototype_similarities * 2.0  # Scale similarity to logit range
                outputs = (1 - self.guidance_weight) * outputs + self.guidance_weight * guidance
                
        return outputs

    def compute_semantic_weights(self, new_labels, all_labels):
        """Compute soft semantic weights for all class pairs"""
        semantic_weights = {}
        
        for class_label in new_labels:
            weights = []
            for ref_class_label in all_labels:
                if class_label != ref_class_label:
                    # Get prototypes
                    if ref_class_label in self.learned_classes:
                        ref_proto = self.op[ref_class_label]
                    else:
                        ref_proto = self.running_op[ref_class_label]
                    
                    # Compute learned similarity
                    sim_score = self.semantic_similarity(
                        self.running_op[class_label], 
                        ref_proto
                    )
                    weights.append(sim_score)
                else:
                    weights.append(torch.tensor(0.0).to(self.device))
            
            semantic_weights[class_label] = torch.stack(weights)
        
        return semantic_weights

    def semantic_contrastive_loss(self, prototypes, semantic_weights, temperature=0.1):
        """Compute contrastive loss with semantic guidance"""
        loss = 0
        
        # Only process new classes that have semantic weights
        new_classes = [c for c in prototypes.keys() if c in semantic_weights]
        
        for class_label in new_classes:
            # Get prototype and semantic weights for this class
            anchor_proto = prototypes[class_label]
            sem_weights = semantic_weights[class_label]
            
            # Compute distances to all other prototypes
            distances = []
            weights = []
            
            for ref_class_label in prototypes.keys():
                if class_label != ref_class_label:
                    dist = F.mse_loss(anchor_proto, prototypes[ref_class_label])
                    distances.append(dist)
                    
                    # Get the corresponding semantic weight for this pair
                    if hasattr(self, 'all_labels_for_weights'):
                        try:
                            weight_idx = self.all_labels_for_weights.index(ref_class_label)
                            if weight_idx < len(sem_weights):
                                weights.append(sem_weights[weight_idx])
                            else:
                                weights.append(torch.tensor(0.5).to(self.device))
                        except (ValueError, IndexError):
                            weights.append(torch.tensor(0.5).to(self.device))
                    else:
                        weights.append(torch.tensor(0.5).to(self.device))
            
            if len(distances) == 0:
                continue
                
            distances = torch.stack(distances)
            weights = torch.stack(weights)
            
            # Ensure dimensions match
            if distances.shape[0] != weights.shape[0]:
                print(f"Warning: Dimension mismatch for class {class_label} - distances: {distances.shape}, weights: {weights.shape}")
                # Use uniform weights as fallback
                weights = torch.ones_like(distances) * 0.5
            
            # Weighted contrastive loss
            # Higher semantic weight = stronger attraction
            weighted_pos_loss = (weights * distances).sum()
            weighted_neg_loss = ((1 - weights) * distances).sum()
            
            if weighted_neg_loss > 0:
                loss += weighted_pos_loss / weighted_neg_loss
        
        return loss

    def update_semantic_similarity(self):
        """Update semantic similarity network using learned similarity from previous tasks"""
        if len(self.learned_classes) < 2:
            return
            
        # Instead of using domain knowledge, use learned similarity from previous tasks
        # This is more fair as it doesn't assume any prior knowledge about class relationships
        
        # Create pairs based on learned similarity scores
        pos_pairs = []
        neg_pairs = []
        
        if len(self.learned_classes) >= 2:
            # Use cosine similarity to find similar/dissimilar pairs (like original SARL)
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            
            # Create similarity matrix for learned classes
            learned_prototypes = []
            learned_class_ids = []
            for class_id in self.learned_classes:
                if class_id in self.op:
                    learned_prototypes.append(self.op[class_id])
                    learned_class_ids.append(class_id)
            
            if len(learned_prototypes) >= 2:
                learned_prototypes = torch.stack(learned_prototypes)
                
                # Compute similarity matrix
                sim_matrix = torch.zeros(len(learned_class_ids), len(learned_class_ids))
                for i in range(len(learned_class_ids)):
                    for j in range(len(learned_class_ids)):
                        if i != j:
                            sim_matrix[i, j] = cos(learned_prototypes[i], learned_prototypes[j])
                
                # Find positive pairs (high similarity) and negative pairs (low similarity)
                threshold = 0.5  # Can be made learnable
                
                for i in range(len(learned_class_ids)):
                    for j in range(i+1, len(learned_class_ids)):
                        if sim_matrix[i, j] > threshold:
                            pos_pairs.append((learned_prototypes[i], learned_prototypes[j]))
                        elif sim_matrix[i, j] < -threshold:
                            neg_pairs.append((learned_prototypes[i], learned_prototypes[j]))
        
        # Train similarity network
        all_pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        
        if len(all_pairs) > 0:
            pair_feats = torch.stack([torch.cat([p1, p2]) for p1, p2 in all_pairs])
            pair_labels = torch.tensor(labels).float().to(self.device)
            
            # Split concatenated features
            feat_dim = self.args.num_feats
            feat1 = pair_feats[:, :feat_dim]
            feat2 = pair_feats[:, feat_dim:]
            
            sim_scores = self.semantic_similarity(feat1, feat2)
            sim_loss = F.binary_cross_entropy(sim_scores, pair_labels)
            
            # Update similarity network
            self.sim_optimizer.zero_grad()
            sim_loss.backward()
            self.sim_optimizer.step()

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        
        if not self.buffer.is_empty():
            # Use balanced sampling if enabled
            if self.args.use_balanced_sampling and isinstance(self.buffer, BalancedBuffer):
                buf_inputs, buf_labels, buf_logits = self.buffer.get_balanced_data(
                    self.args.minibatch_size, 
                    transform=self.transform,
                    op_prototypes=self.op if self.current_task > 0 else None
                )
            else:
                buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, 
                    transform=self.transform
                )
            
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
            reg_loss = self.args.alpha * F.mse_loss(buff_out, buf_logits)

            buff_ce_loss = self.loss(buff_out, buf_labels)
            loss += reg_loss + buff_ce_loss

            # FIXED: Original prototype regularization (no complex multi-centroid)
            if self.current_task > 0:
                buff_feats = F.normalize(buff_feats)
                dist = 0
                for class_label in torch.unique(buf_labels):
                    if class_label in self.learned_classes:
                        image_class_mask = (buf_labels == class_label)
                        mean_feat = buff_feats[image_class_mask].mean(axis=0)
                        dist += F.mse_loss(mean_feat, self.op[class_label])

                loss += self.args.op_weight * dist

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/reg_loss', reg_loss.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/buff_ce_loss', buff_ce_loss.item(), self.iteration)

        outputs, activations = self.net(inputs, return_activations=True)
        current_features = activations['feat']

        # REMOVED: No prototype guidance during training - this was the main issue!

        if self.epoch > self.args.warmup_epochs and self.current_task > 0:
            # Update semantic similarity network
            self.update_semantic_similarity()

            outputs_old = self.net_old(inputs)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

            new_labels = [i.item() for i in torch.unique(labels) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels
            feats = activations['feat']
            feats = F.normalize(feats)

            # Accumulate the class prototypes
            class_prot = {}
            for ref_class_label in self.learned_classes:
                class_prot[ref_class_label] = self.op[ref_class_label]
            for class_label in new_labels:
                class_prot[class_label] = feats[labels == class_label].mean(dim=0)

            # Compute semantic weights
            semantic_weights = self.compute_semantic_weights(new_labels, all_labels)
            
            # Store all_labels for proper indexing in contrastive loss
            self.all_labels_for_weights = all_labels
            
            # Apply semantic-aware contrastive loss
            l_cont = self.semantic_contrastive_loss(class_prot, semantic_weights)
            loss += self.args.sm_weight * l_cont
            
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/contrastive_loss', l_cont.item(), self.iteration)

        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        # Add data to buffer with features for balanced sampling
        if self.args.use_balanced_sampling and isinstance(self.buffer, BalancedBuffer):
            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels[:real_batch_size],
                logits=outputs.data,
                features=activations['feat'][:real_batch_size]
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

        # UNCHANGED: Original prototype calculation (simplified, no complex clustering)
        if self.epoch >= self.args.warmup_epochs and self.eval_prototypes and self.current_task > 0:
            print('!' * 30)
            print('Evaluating Fixed Prototypes for the New Classes')
            # Calculate Class Prototypes
            X = []
            Y = []
            for data in dataset.train_loader:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, activations = self.net(inputs, return_activations=True)
                feat = activations['feat']

                # Normalize Features
                feat = F.normalize(feat)

                X.append(feat.detach().cpu().numpy())
                Y.append(labels.cpu().numpy())

                unique_labels = labels.unique()
                for class_label in unique_labels:
                    self.running_op[class_label] += feat[labels == class_label].sum(dim=0).detach()
                    self.running_sample_counts[class_label] += (labels == class_label).sum().detach()

            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

            # FIXED: High momentum updates for stability
            for class_label in np.unique(Y):
                new_prototype = self.running_op[class_label] / self.running_sample_counts[class_label]

                if class_label in self.learned_classes:
                    # High momentum update for existing prototypes
                    self.op[class_label] = (self.prototype_momentum * self.op[class_label] + 
                                          (1 - self.prototype_momentum) * new_prototype)
                else:
                    # Direct assignment for new classes
                    self.op[class_label] = new_prototype

            # Evaluate semantic relationships using learned similarity
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            print('*' * 30)
            print('Semantic Groups (Learned Similarity - No Domain Knowledge)')
            for class_label in new_labels:
                semantic_scores = []
                for ref_class_label in all_labels:
                    if class_label != ref_class_label:
                        if ref_class_label in self.learned_classes:
                            ref_proto = self.op[ref_class_label]
                        else:
                            ref_proto = self.running_op[ref_class_label]
                        
                        sim_score = self.semantic_similarity(
                            self.running_op[class_label], 
                            ref_proto
                        ).item()
                        semantic_scores.append((ref_class_label, sim_score))
                
                # Sort by similarity and show top similar classes
                semantic_scores.sort(key=lambda x: x[1], reverse=True)
                top_similar = semantic_scores[:5]  # Top 5 similar classes
                
                if self.args.dataset not in ['seq-tinyimg', 'gcil-cifar100']:
                    similar_names = [f"{self.class_dict[c]}({s:.3f})" for c, s in top_similar]
                    print(f'{self.class_dict[class_label]}: ' + ', '.join(similar_names))
                else:
                    similar_classes = [f"{c}({s:.3f})" for c, s in top_similar]
                    print(f'{class_label}: ' + ', '.join(similar_classes))
            print('*' * 30)

            self.eval_prototypes = False

    def end_task(self, dataset) -> None:
        # reset optimizer
        self.get_optimizer()

        self.eval_prototypes = True
        self.flag = True
        self.current_task += 1
        self.net.eval()

        # Save old model
        self.net_old = deepcopy(self.net)
        self.net_old.eval()

        # Buffer Pass
        if isinstance(self.buffer, BalancedBuffer):
            buf_inputs, buf_labels, buf_logits = self.buffer.get_all_data(transform=self.transform)
        else:
            buf_inputs, buf_labels, buf_logits = self.buffer.get_all_data(transform=self.transform)
        
        buf_idx = torch.arange(0, len(buf_labels)).to(buf_labels.device)

        buff_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels, buf_logits, buf_idx)
        buff_data_loader = DataLoader(buff_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        self.net.train()
        for data, label, logits, index in buff_data_loader:
            out_net = self.net(data)

        # =====================================
        # FIXED: Simple prototype calculation with high momentum
        # =====================================
        self.net.eval()
        X = []
        Y = []
        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, activations = self.net(inputs, return_activations=True)
            feat = activations['feat']

            # Normalize Features
            feat = F.normalize(feat)

            X.append(feat.detach().cpu().numpy())
            Y.append(labels.cpu().numpy())

            unique_labels = labels.unique()
            for class_label in unique_labels:
                self.op_sum[class_label] += feat[labels == class_label].sum(dim=0).detach()
                self.sample_counts[class_label] += (labels == class_label).sum().detach()

        X = np.concatenate(X)
        Y = np.concatenate(Y)

        # FIXED: Simple, stable prototype updates
        for class_label in np.unique(Y):
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            
            # Simple average (no complex clustering)
            new_prototype = self.op_sum[class_label] / self.sample_counts[class_label]
            
            # High momentum for stability
            if hasattr(self, 'op') and self.op[class_label].sum() != 0:
                self.op[class_label] = (self.prototype_momentum * self.op[class_label] + 
                                      (1 - self.prototype_momentum) * new_prototype)
            else:
                self.op[class_label] = new_prototype

        # Update buffer with prototypes for balanced sampling
        if isinstance(self.buffer, BalancedBuffer):
            self.buffer.op = self.op

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'fixed_prototypes.pth'))
            torch.save(self.semantic_similarity.state_dict(), os.path.join(model_dir, f'semantic_similarity.pth'))
            torch.save(self.adaptive_threshold.state_dict(), os.path.join(model_dir, f'adaptive_threshold.pth'))

    def get_optimizer(self):
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None 