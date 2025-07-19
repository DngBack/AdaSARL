#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import argparse
from datetime import datetime
import time
import copy
from tqdm import tqdm

from datasets.seq_cifar100 import SequentialCIFAR100
from models.sarl_enhanced_gelu_balanced import SARLEnhancedGeluBalanced, get_parser
from utils.args import *
from utils.best_args import *
from utils.conf import base_path
from utils import create_if_not_exists
from utils.status import create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.buffer import *
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import uuid


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the logits
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                   if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def main():
    # Set up arguments for the original model with CIFAR-100
    args = Namespace(
        # Dataset
        dataset='seq-cifar100',
        N_TASKS=10,
        N_CLASSES_PER_TASK=10,
        
        # Model
        model='sarl_enhanced_gelu_balanced',
        
        # Training
        batch_size=32,
        minibatch_size=32,
        n_epochs=200,
        lr=0.1,
        
        # Buffer
        buffer_size=2000,
        
        # SARL specific
        alpha=0.5,
        beta=1.0,
        op_weight=0.1,
        sm_weight=0.01,
        sim_lr=0.001,
        
        # GELU
        apply_gelu=[1, 1, 1, 1],
        num_feats=512,
        
        # Balanced sampling
        use_balanced_sampling=1,
        balance_weight=1.0,
        
        # Experimental
        save_interim=1,
        warmup_epochs=5,
        use_lr_scheduler=1,
        lr_steps=[70, 90],
        
        # Logging
        csv_log=True,
        tensorboard=True,
        output_dir='./outputs/sarl_enhanced_gelu_balanced_cifar100',
        experiment_id=f'cifar100_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # System
        seed=42,
        gpu=True,
        cuda=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        
        # Disable wandb for simplicity
        nowand=True,
        disable_log=False,
        save_checkpoints=False
    )

    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Create output directory
    create_if_not_exists(args.output_dir)

    # Get dataset
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    
    # Create model
    model = SARLEnhancedGeluBalanced(backbone, loss, args, dataset.get_transform())
    model.net.to(model.device)

    # Set up logging
    if args.csv_log:
        from utils.loggers import CsvLogger
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
        csv_logger.log_arguments(args)
        model.csv_logger = csv_logger

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model.NAME, remove_existing_data=True)
        model.writer = tb_logger.writer
        tb_logger.log_arguments(args)

    print(f"Starting training with model: {model.NAME}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Number of tasks: {dataset.N_TASKS}")
    print(f"Classes per task: {dataset.N_CLASSES_PER_TASK}")

    results, results_mask_classes = [], []

    # Initialize tasks
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset.get_data_loaders()
    if hasattr(model, 'end_task'):
        model.end_task(dataset)

    # Main training loop
    for t in range(dataset.N_TASKS):
        print(f"\n{'='*50}")
        print(f"Training Task {t+1}/{dataset.N_TASKS}")
        print(f"{'='*50}")
        
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
            
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        
        # Training epochs
        for e in range(args.n_epochs):
            if args.tensorboard:
                tb_logger.log_epoch(e, dataset, model)

            train_loss = 0
            train_iter = iter(train_loader)
            
            # Training iterations
            for i in range(len(train_loader)):
                inputs, labels, not_aug_inputs = next(train_iter)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                
                loss = model.observe(inputs, labels, not_aug_inputs)
                assert not np.isnan(loss)
                train_loss += loss

            if scheduler is not None:
                scheduler.step()

            # Print progress every 20 epochs
            if (e + 1) % 20 == 0:
                print(f"Epoch {e+1}/{args.n_epochs}, Loss: {train_loss/len(train_loader):.4f}")

        if hasattr(model, 'end_epoch'):
            model.end_epoch(dataset, e)

        # Evaluate
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs[0])
        print(f"\nTask {t+1} completed. Mean accuracy: {mean_acc:.2f}%")
        
        # Print per-task accuracies
        print("Per-task accuracies:")
        for i, acc in enumerate(accs[0]):
            print(f"  Task {i+1}: {acc:.2f}%")

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

    # Final results
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED")
    print(f"{'='*50}")
    
    final_mean_acc = np.mean(results[-1])
    print(f"Final mean accuracy: {final_mean_acc:.2f}%")
    
    # Calculate forgetting
    if len(results) > 1:
        forgetting = []
        for i in range(len(results) - 1):
            forgetting.append(results[i][i] - results[-1][i])
        avg_forgetting = np.mean(forgetting)
        print(f"Average forgetting: {avg_forgetting:.2f}%")

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        csv_logger.write(vars(args))

    if args.tensorboard:
        tb_logger.close()

    print(f"\nResults saved to: {args.output_dir}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main() 