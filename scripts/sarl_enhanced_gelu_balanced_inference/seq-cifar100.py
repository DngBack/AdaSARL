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
from models.sarl_enhanced_gelu_balanced_inference import SARLEnhancedGeluBalancedInference, get_parser
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
    parser = get_parser()
    args = parser.parse_known_args()[0]
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.csv_log:
        from utils.loggers import CsvLogger
        args.csv_log = False

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = SARLEnhancedGeluBalancedInference(backbone, loss, args, dataset.get_transform())
    model.net.to(model.device)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model.NAME, remove_existing_data=True)

    model.writer = tb_logger.writer
    model.csv_logger = csv_logger
    csv_logger.log_arguments(args)

    if args.csv_log:
        csv_logger.log_arguments(args)
    if args.tensorboard:
        tb_logger.log_arguments(args)

    if args.save_checkpoints:
        print('Creating checkpoint directory...')
        create_if_not_exists(join(args.output_dir, "checkpoints"))

    if torch.cuda.is_available() and args.gpu is True:
        print('Using CUDA')
        args.cuda = True
    elif torch.cuda.is_available() and not args.gpu:
        print('WARNING: You have a CUDA device, so you should probably run with --gpu=True')
    else:
        args.cuda = False

    if args.cuda:
        model.cuda()

    if args.tensorboard:
        tb_logger.writer.add_text("model", str(model))
        tb_logger.writer.add_text("backbone", str(backbone))
        tb_logger.writer.add_text("dataset", str(dataset))
        tb_logger.writer.add_text("args", str(args))

    model.writer = tb_logger.writer
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = get_logger(args, dataset)
        logger.info(args)

    if not args.nowand:
        assert wandb is not None
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        model.wandb = wandb

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset.get_data_loaders()
    if hasattr(model, 'end_task'):
        model.end_task(dataset)

    for t in range(dataset.N_TASKS):
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
        for e in range(args.n_epochs):
            if args.tensorboard:
                tb_logger.log_epoch(e, dataset, model)

            train_loss = 0
            train_acc = 0
            train_iter = iter(train_loader)
            for i in range(len(train_loader)):
                inputs, labels, not_aug_inputs = next(train_iter)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss = model.observe(inputs, labels, not_aug_inputs)
                assert not math.isnan(loss)
                train_loss += loss

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_epoch'):
            model.end_epoch(dataset, e)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs[0])
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                              results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))


if __name__ == '__main__':
    main() 