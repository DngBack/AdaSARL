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
    parser = ArgumentParser(description='Semantic Aware Representation Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sim_thresh', type=float, default=0.80)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    # Sparsity param
    parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--kw', type=float, nargs='*', default=[0.9, 0.9, 0.9, 0.9])
    parser.add_argument('--kw_relu', type=int, default=1)
    parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=512)
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[70, 90])
    
    # Fixed Prototype Parameters (much more conservative)
    parser.add_argument('--prototype_momentum', type=float, default=0.99, help='High momentum for stable updates')
    parser.add_argument('--enable_inference_guidance', type=int, default=1, help='Enable prototype guidance during inference')
    parser.add_argument('--guidance_weight', type=float, default=0.05, help='Weight for prototype guidance (very small)')
    
    # NEW: Contrastive Learning Parameters
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--contrastive_temperature', type=float, default=0.5, help='Temperature for contrastive loss')
    parser.add_argument('--prototype_contrastive_weight', type=float, default=0.3, help='Weight for prototype contrastive loss')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.3, help='Ratio of hard negatives to mine')
    parser.add_argument('--relation_distill_weight', type=float, default=0.2, help='Weight for prototype relation distillation')
    
    return parser


# =============================================================================
# Enhanced Contrastive Loss for SARL
# =============================================================================
class SAR_ContrastiveLoss(nn.Module):
    """
    Enhanced contrastive loss that combines:
    1. Instance-to-instance contrastive learning
    2. Instance-to-prototype contrastive learning  
    3. Semantic-aware hard negative mining
    4. Prototype-instance relation distillation
    """
    def __init__(self, temperature=0.5, prototype_weight=0.3, hard_negative_ratio=0.3):
        super().__init__()
        self.temperature = temperature
        self.prototype_weight = prototype_weight
        self.hard_negative_ratio = hard_negative_ratio
        
    def forward(self, features, labels, prototypes, learned_classes, pos_groups, 
                buffer_features=None, buffer_labels=None):
        """
        Args:
            features: Current batch features [B, D]
            labels: Current batch labels [B]
            prototypes: Class prototypes [C, D]
            learned_classes: List of learned class indices
            pos_groups: Dict mapping class to semantically similar classes
            buffer_features: Buffer features for hard negative mining [B_buf, D]
            buffer_labels: Buffer labels [B_buf]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize all features and prototypes
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        if buffer_features is not None:
            buffer_features = F.normalize(buffer_features, dim=1)
        
        total_loss = 0
        
        # 1. Instance-to-Instance Contrastive Loss
        instance_loss = self._instance_contrastive_loss(features, labels)
        total_loss += instance_loss
        
        # 2. Instance-to-Prototype Contrastive Loss (for learned classes)
        if len(learned_classes) > 0:
            prototype_loss = self._prototype_contrastive_loss(
                features, labels, prototypes, learned_classes
            )
            total_loss += self.prototype_weight * prototype_loss
        
        # 3. Semantic-Aware Contrastive Loss (using pos_groups)
        if len(learned_classes) > 0 and pos_groups:
            semantic_loss = self._semantic_aware_contrastive_loss(
                features, labels, prototypes, pos_groups, learned_classes
            )
            total_loss += semantic_loss
            
        # 4. Hard Negative Mining (using buffer)
        if buffer_features is not None and len(buffer_features) > 0:
            hard_neg_loss = self._hard_negative_mining_loss(
                features, labels, buffer_features, buffer_labels
            )
            total_loss += hard_neg_loss
            
        return total_loss
    
    def _instance_contrastive_loss(self, features, labels):
        """Standard instance-to-instance contrastive loss with numerical safeguards"""
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Create positive/negative masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Remove self-comparison
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Check if any sample has positive pairs
        mask_sum = mask.sum(1)
        if mask_sum.min() == 0:
            # If some samples have no positive pairs, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_denominator = torch.log(torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-8))
        log_prob = logits - log_denominator
        
        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask_sum, min=1e-8)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return torch.clamp(loss, max=10.0)  # Clip to prevent extreme values
    
    def _prototype_contrastive_loss(self, features, labels, prototypes, learned_classes):
        """Instance-to-prototype contrastive loss with numerical safeguards"""
        if len(learned_classes) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        # Compute similarities to all learned prototypes
        learned_prototypes = prototypes[learned_classes]
        similarities = torch.matmul(features, learned_prototypes.T) / self.temperature
        
        # Create target mask for positive prototypes
        targets = torch.zeros(len(features), len(learned_classes)).to(features.device)
        for i, label in enumerate(labels):
            if label.item() in learned_classes:
                class_idx = learned_classes.index(label.item())
                targets[i, class_idx] = 1.0
        
        # Check if we have any positive targets
        if targets.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Compute cross-entropy with prototypes
        log_probs = F.log_softmax(similarities, dim=1)
        loss = -(targets * log_probs).sum() / torch.clamp(targets.sum(), min=1e-8)
        
        return torch.clamp(loss, max=10.0)
    
    def _semantic_aware_contrastive_loss(self, features, labels, prototypes, pos_groups, learned_classes):
        """Semantic-aware contrastive loss using positive groups"""
        total_loss = 0
        count = 0
        
        for class_label in torch.unique(labels):
            class_label_item = class_label.item()
            if class_label_item not in learned_classes or class_label_item not in pos_groups:
                continue
                
            class_mask = (labels == class_label)
            class_features = features[class_mask]
            
            if len(class_features) == 0:
                continue
            
            # Get positive and negative classes based on semantic groups
            pos_classes = [c for c in pos_groups[class_label_item] if c in learned_classes]
            all_learned = set(learned_classes)
            neg_classes = list(all_learned - set(pos_classes) - {class_label_item})
            
            if len(pos_classes) > 0 and len(neg_classes) > 0:
                # Positive similarities (including self)
                pos_prototypes = prototypes[[class_label_item] + pos_classes]
                pos_sim = torch.matmul(class_features, pos_prototypes.T) / self.temperature
                
                # Negative similarities
                neg_prototypes = prototypes[neg_classes]
                neg_sim = torch.matmul(class_features, neg_prototypes.T) / self.temperature
                
                # Combine and compute contrastive loss
                all_sim = torch.cat([pos_sim, neg_sim], dim=1)
                pos_targets = torch.zeros(len(class_features), all_sim.shape[1]).to(features.device)
                pos_targets[:, :len(pos_prototypes)] = 1.0 / len(pos_prototypes)  # Uniform over positives
                
                log_probs = F.log_softmax(all_sim, dim=1)
                loss = -(pos_targets * log_probs).sum() / torch.clamp(pos_targets.sum(), min=1e-8)
                
                # Clamp loss to prevent extreme values
                loss = torch.clamp(loss, max=10.0)
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def _hard_negative_mining_loss(self, features, labels, buffer_features, buffer_labels):
        """Hard negative mining using buffer samples with numerical safeguards"""
        if buffer_features is None or len(buffer_features) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        batch_size = features.shape[0]
        buffer_size = buffer_features.shape[0]
        
        # Compute similarities between current features and buffer features
        similarities = torch.matmul(features, buffer_features.T)  # [B, B_buf]
        
        total_loss = 0
        count = 0
        
        for i, label in enumerate(labels):
            # Find hard negatives: buffer samples with different labels but high similarity
            different_label_mask = (buffer_labels != label)
            if different_label_mask.sum() == 0:
                continue
                
            # Get similarities to different-label buffer samples
            neg_similarities = similarities[i][different_label_mask]
            
            if len(neg_similarities) == 0:
                continue
                
            # Mine hard negatives (top similarities)
            num_hard_neg = max(1, int(self.hard_negative_ratio * len(neg_similarities)))
            hard_neg_sim, _ = torch.topk(neg_similarities, min(num_hard_neg, len(neg_similarities)))
            
            # Find positive samples: buffer samples with same label
            same_label_mask = (buffer_labels == label)
            if same_label_mask.sum() > 0:
                pos_similarities = similarities[i][same_label_mask]
                
                # Contrastive loss: pull positives, push hard negatives
                # Clamp to prevent overflow
                pos_scaled = torch.clamp(pos_similarities / self.temperature, max=50.0)
                neg_scaled = torch.clamp(hard_neg_sim / self.temperature, max=50.0)
                
                pos_term = torch.logsumexp(pos_scaled, dim=0)
                neg_term = torch.logsumexp(neg_scaled, dim=0)
                
                loss = neg_term - pos_term
                loss = torch.clamp(loss, max=10.0)  # Clamp final loss
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)


class PrototypeRelationDistillation(nn.Module):
    """
    Maintains prototype-instance relationships across tasks using knowledge distillation
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, old_prototypes, new_prototypes, learned_classes):
        """
        Args:
            features: Current features [B, D]
            old_prototypes: Previous task prototypes [C, D]
            new_prototypes: Current prototypes [C, D]  
            learned_classes: List of previously learned classes
        """
        if len(learned_classes) == 0:
            return torch.tensor(0.0).to(features.device)
            
        features = F.normalize(features, dim=1)
        old_prototypes = F.normalize(old_prototypes, dim=1)
        new_prototypes = F.normalize(new_prototypes, dim=1)
        
        # Select prototypes for learned classes only
        old_learned_prototypes = old_prototypes[learned_classes]
        new_learned_prototypes = new_prototypes[learned_classes]
        
        # Compute prototype-instance similarities
        old_similarities = torch.matmul(features, old_learned_prototypes.T) / self.temperature
        new_similarities = torch.matmul(features, new_learned_prototypes.T) / self.temperature
        
        # Distillation loss: maintain the relative relationships
        old_relations = F.softmax(old_similarities, dim=1)
        new_relations = F.log_softmax(new_similarities, dim=1)
        
        # KL divergence loss
        distill_loss = F.kl_div(new_relations, old_relations, reduction='batchmean')
        
        return distill_loss


# =============================================================================
# Enhanced SARL with Contrastive Learning
# =============================================================================
class SARL(ContinualModel):
    NAME = 'sarl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        # Call parent constructor first (creates self.net = backbone and self.opt)
        super(SARL, self).__init__(backbone, loss, args, transform)
        
        # Override with our custom network (following SARLOnline pattern)
        if 'mnist' in args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(args.device)
        else:
            self.net = sparse_resnet18(
                nclasses=num_classes_dict[args.dataset],
                kw_percent_on=args.kw, local=args.kw_local,
                relu=args.kw_relu, apply_kw=args.apply_kw
            ).to(args.device)
        
        # Override optimizer to use our custom network
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        
        # Initialize scheduler
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
        
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.net_old = None

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

        # NEW: Contrastive Learning Components
        self.contrastive_loss_fn = SAR_ContrastiveLoss(
            temperature=args.contrastive_temperature,
            prototype_weight=args.prototype_contrastive_weight,
            hard_negative_ratio=args.hard_negative_ratio
        )
        self.relation_distill_fn = PrototypeRelationDistillation()
        self.contrastive_weight = args.contrastive_weight
        self.relation_distill_weight = args.relation_distill_weight
        
        # Store old prototypes for distillation
        self.op_old = None

        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes_dict[args.dataset], num_classes_dict[args.dataset]).to(self.device)
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

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        
        # Buffer samples and features for contrastive learning
        buf_features = None
        buf_labels = None
        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
            buf_features = buff_feats.detach()  # Store for contrastive learning
            
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

            # Original SARL semantic contrastive loss
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
            
            # NEW: Enhanced Contrastive Learning
            if len(self.learned_classes) > 0:
                # Enhanced contrastive loss
                contrastive_loss = self.contrastive_loss_fn(
                    features=current_features,
                    labels=labels,
                    prototypes=self.op,
                    learned_classes=self.learned_classes,
                    pos_groups=self.pos_groups,
                    buffer_features=buf_features,
                    buffer_labels=buf_labels
                )
                loss += self.contrastive_weight * contrastive_loss
                
                # Prototype relation distillation (if we have old prototypes)
                if self.op_old is not None:
                    relation_loss = self.relation_distill_fn(
                        features=current_features,
                        old_prototypes=self.op_old,
                        new_prototypes=self.op,
                        learned_classes=self.learned_classes
                    )
                    loss += self.relation_distill_weight * relation_loss
                    
                    if hasattr(self, 'writer'):
                        self.writer.add_scalar(f'Task {self.current_task}/contrastive_loss', contrastive_loss.item(), self.iteration)
                        self.writer.add_scalar(f'Task {self.current_task}/relation_distill_loss', relation_loss.item(), self.iteration)

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/original_contrastive_loss', l_cont.item(), self.iteration)

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

            # Evaluate the distances
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)

            # dist_mat = torch.zeros(len(all_labels), len(all_labels))
            for class_label in new_labels:
                for ref_class_label in new_labels:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.running_op[ref_class_label])
                for ref_class_label in self.learned_classes:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.op[ref_class_label])

            print('*' * 30)
            print('Positive Groups')
            for class_label in new_labels:
                pos_group = self.dist_mat[class_label] > self.args.sim_thresh
                self.pos_groups[class_label] = [i for i in all_labels if pos_group[i]]
                print(f'{class_label}:', self.pos_groups[class_label])
            print('*' * 30)
            self.eval_prototypes = False

    def end_task(self, dataset) -> None:

        # reset optimizer
        self.get_optimizer()

        self.eval_prototypes = True
        self.flag = True
        self.current_task += 1
        self.net.eval()

        # Save old model and prototypes
        self.net_old = deepcopy(self.net)
        self.net_old.eval()
        
        # NEW: Save old prototypes for relation distillation
        self.op_old = self.op.clone().detach()

        # =====================================
        # Buffer Pass
        # =====================================
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

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'fixed_prototypes.ph'))

    def get_optimizer(self):
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
