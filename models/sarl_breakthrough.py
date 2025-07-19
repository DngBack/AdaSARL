import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import math

class OutOfTaskDetection(nn.Module):
    """ARC-inspired Out-of-Task Detection for test-time adaptation"""
    
    def __init__(self, feature_dim: int, num_tasks: int, confidence_threshold: float = 0.8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.confidence_threshold = confidence_threshold
        self.task_prototypes = nn.Parameter(torch.randn(num_tasks, feature_dim))
        self.task_counters = torch.zeros(num_tasks)
        
    def update_prototypes(self, features: torch.Tensor, task_id: int):
        """Update task prototypes with new features"""
        with torch.no_grad():
            if self.task_counters[task_id] == 0:
                self.task_prototypes[task_id] = features.mean(dim=0)
            else:
                # Exponential moving average
                alpha = 0.1
                self.task_prototypes[task_id] = (1 - alpha) * self.task_prototypes[task_id] + alpha * features.mean(dim=0)
            self.task_counters[task_id] += 1
    
    def detect_task(self, features: torch.Tensor) -> Tuple[int, float]:
        """Detect which task the features belong to"""
        similarities = F.cosine_similarity(features.unsqueeze(1), self.task_prototypes.unsqueeze(0), dim=2)
        max_sim, predicted_task = torch.max(similarities, dim=1)
        confidence = torch.sigmoid(max_sim * 5)  # Scale for better confidence
        
        return predicted_task.item(), confidence.mean().item()

class AdaptiveRetentionCorrection(nn.Module):
    """ARC-inspired Adaptive Retention & Correction mechanism"""
    
    def __init__(self, classifier: nn.Module, feature_extractor: nn.Module, num_classes: int):
        super().__init__()
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.original_classifier = copy.deepcopy(classifier)
        self.task_boundaries = []
        self.current_task = 0
        
    def adaptive_retention(self, features: torch.Tensor, task_id: int, lr: float = 0.01):
        """Dynamically tune classifier for past task data"""
        # Create pseudo-labels for retention
        with torch.no_grad():
            original_logits = self.original_classifier(features)
            pseudo_labels = torch.argmax(original_logits, dim=1)
        
        # Fine-tune classifier on pseudo-labeled data
        self.classifier.train()
        current_logits = self.classifier(features)
        retention_loss = F.cross_entropy(current_logits, pseudo_labels)
        
        # Gradient update
        for param in self.classifier.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        retention_loss.backward(retain_graph=True)
        
        with torch.no_grad():
            for param in self.classifier.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
        
        return retention_loss.item()
    
    def adaptive_correction(self, logits: torch.Tensor, task_id: int) -> torch.Tensor:
        """Correct predictions for past task data"""
        corrected_logits = logits.clone()
        
        # Boost confidence for classes from detected task
        if task_id < self.current_task:
            start_idx = sum(self.task_boundaries[:task_id])
            end_idx = start_idx + self.task_boundaries[task_id]
            
            # Apply temperature scaling for past tasks
            temp = 0.8  # Lower temperature = higher confidence
            corrected_logits[:, start_idx:end_idx] = corrected_logits[:, start_idx:end_idx] / temp
        
        return corrected_logits

class GradientCoresetReplay(nn.Module):
    """GCR-inspired Gradient-based Coreset Selection"""
    
    def __init__(self, buffer_size: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.buffer_inputs = []
        self.buffer_labels = []
        self.buffer_logits = []
        self.buffer_gradients = []
        
    def compute_gradient_norm(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute gradient norm for importance estimation"""
        model.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        
        return math.sqrt(grad_norm)
    
    def select_coreset(self, model: nn.Module, new_inputs: torch.Tensor, new_labels: torch.Tensor, 
                      new_logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select k samples based on gradient-based importance"""
        importances = []
        
        for i in range(len(new_inputs)):
            importance = self.compute_gradient_norm(model, new_inputs[i:i+1], new_labels[i:i+1])
            importances.append(importance)
        
        # Select top-k samples
        indices = torch.topk(torch.tensor(importances), k, largest=True).indices
        
        return new_inputs[indices], new_labels[indices], new_logits[indices]
    
    def update_buffer(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
        """Update buffer with gradient-based selection"""
        if len(self.buffer_inputs) < self.buffer_size:
            # Buffer not full, add all
            self.buffer_inputs.append(inputs)
            self.buffer_labels.append(labels)
            self.buffer_logits.append(logits)
        else:
            # Buffer full, use gradient-based selection
            combined_inputs = torch.cat(self.buffer_inputs + [inputs], dim=0)
            combined_labels = torch.cat(self.buffer_labels + [labels], dim=0)
            combined_logits = torch.cat(self.buffer_logits + [logits], dim=0)
            
            # Select best samples
            selected_inputs, selected_labels, selected_logits = self.select_coreset(
                model, combined_inputs, combined_labels, combined_logits, self.buffer_size
            )
            
            self.buffer_inputs = [selected_inputs]
            self.buffer_labels = [selected_labels]
            self.buffer_logits = [selected_logits]
    
    def get_buffer_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get current buffer data"""
        if not self.buffer_inputs:
            return None, None, None
        
        return (torch.cat(self.buffer_inputs, dim=0),
                torch.cat(self.buffer_labels, dim=0),
                torch.cat(self.buffer_logits, dim=0))

class COREAdaptiveSelection(nn.Module):
    """CORE-inspired Adaptive Quantity Allocation and Quality-Focused Data Selection"""
    
    def __init__(self, buffer_size: int, num_classes: int, window_size: int = 10):
        super().__init__()
        self.buffer_size = buffer_size
        self.num_classes = num_classes
        self.window_size = window_size
        self.forgetting_rates = defaultdict(list)
        self.class_allocations = defaultdict(int)
        self.quality_scores = defaultdict(list)
        
    def update_forgetting_rate(self, class_id: int, accuracy: float):
        """Update forgetting rate for a class"""
        self.forgetting_rates[class_id].append(1 - accuracy)
        if len(self.forgetting_rates[class_id]) > self.window_size:
            self.forgetting_rates[class_id].pop(0)
    
    def get_adaptive_allocation(self) -> Dict[int, int]:
        """Get adaptive allocation based on forgetting rates"""
        total_forgetting = 0
        for class_id in range(self.num_classes):
            if self.forgetting_rates[class_id]:
                total_forgetting += np.mean(self.forgetting_rates[class_id])
        
        allocations = {}
        for class_id in range(self.num_classes):
            if self.forgetting_rates[class_id] and total_forgetting > 0:
                forgetting_rate = np.mean(self.forgetting_rates[class_id])
                allocation = int(self.buffer_size * (forgetting_rate / total_forgetting))
                allocations[class_id] = max(1, allocation)  # At least 1 sample
            else:
                allocations[class_id] = 1
        
        return allocations
    
    def quality_selection(self, inputs: torch.Tensor, labels: torch.Tensor, 
                         features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select k highest quality samples"""
        # Quality based on feature diversity and representativeness
        quality_scores = []
        
        for i in range(len(inputs)):
            # Distance to class centroid (lower is better)
            class_mask = labels == labels[i]
            class_features = features[class_mask]
            centroid = class_features.mean(dim=0)
            dist_to_centroid = F.cosine_similarity(features[i], centroid, dim=0)
            
            # Diversity score (higher is better)
            other_features = features[torch.arange(len(features)) != i]
            if len(other_features) > 0:
                diversity = F.cosine_similarity(features[i].unsqueeze(0), other_features, dim=1).mean()
                diversity = 1 - diversity  # Higher diversity = lower similarity
            else:
                diversity = 1.0
            
            quality = dist_to_centroid * 0.7 + diversity * 0.3
            quality_scores.append(quality.item())
        
        # Select top-k samples
        indices = torch.topk(torch.tensor(quality_scores), k, largest=True).indices
        return inputs[indices], labels[indices]

class RefreshLearning:
    """Neuroscience-inspired Refresh Learning"""
    
    def __init__(self, unlearn_steps: int = 3, relearn_steps: int = 5):
        self.unlearn_steps = unlearn_steps
        self.relearn_steps = relearn_steps
        self.original_state = None
        
    def refresh_on_batch(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, 
                        optimizer: torch.optim.Optimizer, lr: float = 0.01):
        """Perform refresh learning on a batch"""
        # Store original state
        self.original_state = copy.deepcopy(model.state_dict())
        
        # Unlearning phase - minimize negative loss
        for _ in range(self.unlearn_steps):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = -F.cross_entropy(logits, labels)  # Negative loss for unlearning
            loss.backward()
            optimizer.step()
        
        # Relearning phase - normal learning
        for _ in range(self.relearn_steps):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        
        return loss.item()

class TestTimeFinetuning:
    """Test-Time Fine-Tuning mechanism"""
    
    def __init__(self, adaptation_steps: int = 1, adaptation_lr: float = 0.001):
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.original_state = None
        
    def adapt_to_sample(self, model: nn.Module, inputs: torch.Tensor, confidence_threshold: float = 0.7):
        """Adapt model to test sample if confidence is low"""
        model.eval()
        
        # Store original state
        self.original_state = copy.deepcopy(model.state_dict())
        
        with torch.no_grad():
            logits = model(inputs)
            confidence = F.softmax(logits, dim=1).max(dim=1)[0]
        
        # Adapt only if confidence is low
        if confidence.mean() < confidence_threshold:
            model.train()
            
            # Create augmented versions
            augmented_inputs = []
            for _ in range(8):  # 8 augmentations
                # Simple augmentation - add noise
                noise = torch.randn_like(inputs) * 0.1
                augmented_inputs.append(inputs + noise)
            
            augmented_inputs = torch.cat(augmented_inputs, dim=0)
            
            # Entropy minimization
            optimizer = torch.optim.Adam(model.parameters(), lr=self.adaptation_lr)
            
            for _ in range(self.adaptation_steps):
                optimizer.zero_grad()
                aug_logits = model(augmented_inputs)
                
                # Entropy loss
                probs = F.softmax(aug_logits, dim=1)
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                
                entropy_loss.backward()
                optimizer.step()
        
        model.eval()
        return model(inputs)
    
    def reset_model(self, model: nn.Module):
        """Reset model to original state"""
        if self.original_state is not None:
            model.load_state_dict(self.original_state)

class SARLBreakthrough(nn.Module):
    """Breakthrough SARL with latest 2024-2025 techniques"""
    
    def __init__(self, backbone, num_classes, feature_dim=512, buffer_size=500, 
                 num_tasks=5, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self.num_tasks = num_tasks
        self.device = device
        self.current_task = 0
        self.classes_per_task = num_classes // num_tasks
        
        # Core components
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Breakthrough components
        self.otd = OutOfTaskDetection(feature_dim, num_tasks)
        self.arc = AdaptiveRetentionCorrection(self.classifier, self.backbone, num_classes)
        self.gcr = GradientCoresetReplay(buffer_size, feature_dim, num_classes)
        self.core = COREAdaptiveSelection(buffer_size, num_classes)
        self.refresh = RefreshLearning()
        self.ttft = TestTimeFinetuning()
        
        # Task-specific parameters
        self.task_boundaries = []
        self.task_prototypes = {}
        self.class_prototypes = {}
        self.adaptation_history = []
        
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return features
    
    def begin_task(self, task_id, num_classes_in_task):
        """Initialize new task"""
        self.current_task = task_id
        self.task_boundaries.append(num_classes_in_task)
        
        # Update ARC component
        self.arc.current_task = task_id
        self.arc.task_boundaries = self.task_boundaries
        
        print(f"Beginning task {task_id} with {num_classes_in_task} classes")
    
    def observe(self, inputs, labels, logits=None, task_id=None):
        """Observe new data with breakthrough techniques"""
        if logits is None:
            with torch.no_grad():
                logits = self.forward(inputs)
        
        # Extract features
        features = self.extract_features(inputs)
        
        # Update task prototypes for OTD
        if task_id is not None:
            self.otd.update_prototypes(features, task_id)
        
        # Update class prototypes
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in self.class_prototypes:
                self.class_prototypes[label_item] = features[i].clone()
            else:
                # Exponential moving average
                alpha = 0.1
                self.class_prototypes[label_item] = (1 - alpha) * self.class_prototypes[label_item] + alpha * features[i]
        
        # Update CORE forgetting rates
        if task_id is not None and task_id < self.current_task:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).float().mean().item()
                for label in labels:
                    self.core.update_forgetting_rate(label.item(), accuracy)
        
        # Update GCR buffer
        self.gcr.update_buffer(self, inputs, labels, logits)
        
        # Store adaptation history
        self.adaptation_history.append({
            'task_id': task_id,
            'features': features.detach().cpu(),
            'labels': labels.cpu(),
            'logits': logits.detach().cpu()
        })
        
        # Keep only recent history
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
    
    def predict_with_adaptation(self, inputs):
        """Predict with test-time adaptation"""
        # First, get basic prediction
        features = self.extract_features(inputs)
        
        # Detect task using OTD
        task_id, confidence = self.otd.detect_task(features)
        
        # Apply test-time fine-tuning if needed
        if confidence < 0.8:
            logits = self.ttft.adapt_to_sample(self, inputs, confidence_threshold=0.7)
        else:
            logits = self.forward(inputs)
        
        # Apply ARC correction
        corrected_logits = self.arc.adaptive_correction(logits, task_id)
        
        # Apply adaptive retention if needed
        if task_id < self.current_task:
            retention_loss = self.arc.adaptive_retention(features, task_id)
            # Re-predict after retention
            corrected_logits = self.forward(inputs)
            corrected_logits = self.arc.adaptive_correction(corrected_logits, task_id)
        
        # Reset TTFT model
        self.ttft.reset_model(self)
        
        return corrected_logits
    
    def get_buffer_data(self):
        """Get buffer data for replay"""
        return self.gcr.get_buffer_data()
    
    def refresh_learning_step(self, inputs, labels, optimizer):
        """Perform refresh learning step"""
        return self.refresh.refresh_on_batch(self, inputs, labels, optimizer)
    
    def compute_distillation_loss(self, old_logits, new_logits, temperature=4.0):
        """Compute knowledge distillation loss"""
        old_probs = F.softmax(old_logits / temperature, dim=1)
        new_log_probs = F.log_softmax(new_logits / temperature, dim=1)
        return F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (temperature ** 2)
    
    def compute_prototype_loss(self, features, labels, margin=1.0):
        """Compute prototype-based contrastive loss"""
        loss = 0.0
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item in self.class_prototypes:
                # Positive distance (should be small)
                pos_dist = F.cosine_similarity(features[i], self.class_prototypes[label_item], dim=0)
                
                # Negative distances (should be large)
                neg_dists = []
                for other_label, other_prototype in self.class_prototypes.items():
                    if other_label != label_item:
                        neg_dist = F.cosine_similarity(features[i], other_prototype, dim=0)
                        neg_dists.append(neg_dist)
                
                if neg_dists:
                    neg_dist = torch.stack(neg_dists).max()
                    # Margin-based loss
                    loss += F.relu(margin - pos_dist + neg_dist)
        
        return loss / len(labels) if len(labels) > 0 else torch.tensor(0.0, device=features.device) 