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
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18 import sparse_resnet18
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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Enhanced Semantic Aware Representation Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    parser.add_argument('--sim_lr', type=float, default=0.001)
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
    return parser


class SARLEnhanced(ContinualModel):
    NAME = 'sarl_enhanced'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLEnhanced, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        if 'mnist' in self.args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(self.device)
        else:
            self.net = sparse_resnet18(
                nclasses=num_classes_dict[args.dataset],
                kw_percent_on=args.kw, local=args.kw_local,
                relu=args.kw_relu, apply_kw=args.apply_kw
            ).to(self.device)

        self.net_old = None
        self.get_optimizer()

        # set regularization weight
        self.alpha = args.alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        # init Object Prototypes
        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.op_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        
        # Enhanced semantic learning components
        self.semantic_similarity = SemanticSimilarity(args.num_feats).to(self.device)
        self.adaptive_threshold = AdaptiveThreshold().to(self.device)
        self.semantic_weights = {}
        self.sim_optimizer = Adam(self.semantic_similarity.parameters(), lr=args.sim_lr)
        self.threshold_optimizer = Adam(self.adaptive_threshold.parameters(), lr=args.sim_lr)
        
        self.class_dict = class_dict[args.dataset]

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
        """Update semantic similarity network using current prototypes"""
        if len(self.learned_classes) < 2:
            return
            
        # Create positive pairs from similar classes (using domain knowledge)
        pos_pairs = self.create_positive_pairs()
        
        # Create negative pairs from dissimilar classes
        neg_pairs = self.create_negative_pairs()
        
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

    def create_positive_pairs(self):
        """Create positive pairs for semantic similarity training"""
        pos_pairs = []
        
        # Use domain knowledge for CIFAR-10
        if self.args.dataset == 'seq-cifar10':
            # Group similar classes based on semantic categories
            vehicle_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
            animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
            
            # Create pairs within categories
            for category in [vehicle_classes, animal_classes]:
                category = [c for c in category if c in self.learned_classes]
                for i in range(len(category)):
                    for j in range(i+1, len(category)):
                        if category[i] in self.op and category[j] in self.op:
                            pos_pairs.append((self.op[category[i]], self.op[category[j]]))
        
        # Use domain knowledge for CIFAR-100
        elif self.args.dataset == 'seq-cifar100':
            # Group similar classes based on semantic categories
            animal_classes = [0, 1, 2, 3, 4, 6, 7, 8, 9, 14, 15, 16, 18, 19, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
            vehicle_classes = [1, 8, 13, 48, 58, 81, 89, 90]
            plant_classes = [47, 52, 56, 59, 60, 62, 70, 82, 83, 92, 96]
            
            # Create pairs within categories
            for category in [animal_classes, vehicle_classes, plant_classes]:
                category = [c for c in category if c in self.learned_classes]
                for i in range(len(category)):
                    for j in range(i+1, len(category)):
                        if category[i] in self.op and category[j] in self.op:
                            pos_pairs.append((self.op[category[i]], self.op[category[j]]))
        
        return pos_pairs

    def create_negative_pairs(self):
        """Create negative pairs for semantic similarity training"""
        neg_pairs = []
        
        if len(self.learned_classes) < 2:
            return neg_pairs
            
        # Create pairs between different categories for CIFAR-10
        if self.args.dataset == 'seq-cifar10':
            vehicle_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
            animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
            
            # Create cross-category pairs
            for vehicle in vehicle_classes:
                for animal in animal_classes:
                    if vehicle in self.op and animal in self.op:
                        neg_pairs.append((self.op[vehicle], self.op[animal]))
        
        # Create pairs between different categories for CIFAR-100
        elif self.args.dataset == 'seq-cifar100':
            animal_classes = [0, 2, 3, 4, 6, 7, 9, 14, 15, 16, 18, 19, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
            vehicle_classes = [1, 8, 13, 48, 58, 81, 89, 90]
            
            # Create cross-category pairs
            for animal in animal_classes[:5]:  # Limit to avoid too many pairs
                for vehicle in vehicle_classes[:3]:
                    if animal in self.op and vehicle in self.op:
                        neg_pairs.append((self.op[animal], self.op[vehicle]))
        
        return neg_pairs

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
            reg_loss = self.args.alpha * F.mse_loss(buff_out, buf_logits)

            buff_ce_loss = self.loss(buff_out, buf_labels)
            loss += reg_loss + buff_ce_loss

            # Regularization loss on Class Prototypes
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

        # Calculate the Class Prototypes and Covariance Matrices using Working Model
        if self.epoch >= self.args.warmup_epochs and self.eval_prototypes and self.current_task > 0:
            print('!' * 30)
            print('Evaluating Prototypes for the New Classes')
            
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

            # Take average feats
            for class_label in np.unique(Y):
                self.running_op[class_label] = self.running_op[class_label] / self.running_sample_counts[class_label]

            # Evaluate semantic relationships using learned similarity
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            print('*' * 30)
            print('Semantic Groups (Learned Similarity)')
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
        buf_inputs, buf_labels, buf_logits = self.buffer.get_all_data(transform=self.transform)
        buf_idx = torch.arange(0, len(buf_labels)).to(buf_labels.device)

        buff_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels, buf_logits, buf_idx)
        buff_data_loader = DataLoader(buff_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        self.net.train()
        for data, label, logits, index in buff_data_loader:
            out_net = self.net(data)

        # Calculate Class Prototypes
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

        # Take average feats
        for class_label in np.unique(Y):
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            self.op[class_label] = self.op_sum[class_label] / self.sample_counts[class_label]

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'object_prototypes.pth'))
            torch.save(self.semantic_similarity.state_dict(), os.path.join(model_dir, f'semantic_similarity.pth'))

    def get_optimizer(self):
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None 