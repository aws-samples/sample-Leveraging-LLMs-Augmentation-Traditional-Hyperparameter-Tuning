#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################
"""
Culled version with only PyTorch training and plotting functionality
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import logging
import shutil
import sys
import importlib
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_empty_log_directory(log_dir="./logs"):
    """
    Ensures the specified log directory is empty by removing all files in it.
    If the directory doesn't exist, it creates it.
    """
    try:
        # Check if directory exists
        if os.path.exists(log_dir):
            # If it exists, check if it's a directory
            if os.path.isdir(log_dir):
                # Remove all files individually
                for filename in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            print(f"{e}")
                            continue
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"All files removed from {log_dir} directory.")
            else:
                # If it's not a directory but a file, remove it and create directory
                os.unlink(log_dir)
                os.makedirs(log_dir)
                print(f"Removed file named {log_dir} and created directory instead.")
        else:
            # If it doesn't exist, create it
            os.makedirs(log_dir)
            print(f"Created new {log_dir} directory.")

        return True
    except Exception as e:
        print(f"Error ensuring empty log directory: {e}")
        return False

# Set up argument parser for hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description='CNN Training with Gradient Tracking')

    # Model architecture
    parser.add_argument('--num_conv_layers', type=int, default=10, help='Number of convolutional layers')
    parser.add_argument('--filters', type=str, default='64,128,256,512,512,512,512,512,512,512', help='Number of filters per layer, comma-separated')
    parser.add_argument('--kernel_size', type=int, default=15, help='Kernel size for convolutions')
    parser.add_argument('--fc_dims', type=str, default='2,1', help='Dimensions of fully connected layers, comma-separated')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for FC layers')
    parser.add_argument('--use_skip_connections', action='store_true', help='Use skip connections in architecture')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--use_LLM_config', action='store_false', help='Use LLM generated optimization config.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=512, help='Input batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--lr_steps', type=str, default='30,60,80', help='Epochs at which to reduce LR (for step schedule)')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR reduction factor')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'RMSprop'], help='Optimizer')
    parser.add_argument('--gradient_clip', type=float, default=None, help='Gradient clipping value')

    # Data and infrastructure
    parser.add_argument('--data_path', type=str, default='./data/imagenet', help='Path to ImageNet data')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')

    args = parser.parse_args()

    # Process string arguments into lists
    args.filters = [int(f) for f in args.filters.split(',')]
    args.fc_dims = [int(d) for d in args.fc_dims.split(',')]
    args.lr_steps = [int(s) for s in args.lr_steps.split(',')]

    return args

# Function to calculate gradient norm for each layer and total
def get_gradient_norms(model):
    total_norm = 0
    layer_norms = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            layer_norms[name] = param_norm.item()
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm, layer_norms

# Setup data transforms and loaders
def setup_data(args, rank, world_size, quick_test=True, num_samples=100, seed=42):
    """
    Set up data loaders with option for quick testing with reduced dataset
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load dataset (using CIFAR-10)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)

    # If quick test is enabled, select only a subset of images
    if quick_test:
        # Create random indices but make sure we get some from each class for balance
        train_indices = []
        val_indices = []

        # CIFAR-10 has 10 classes, try to get even representation
        samples_per_class = max(num_samples // 10, 1)  # at least 1 sample per class

        # Get indices for each class in training set
        train_targets = np.array(train_dataset.targets)
        for class_idx in range(10):
            class_indices = np.where(train_targets == class_idx)[0]
            # Randomly select samples_per_class indices for this class
            if len(class_indices) > samples_per_class:
                selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                train_indices.extend(selected_indices)
            else:
                train_indices.extend(class_indices)

        # Get indices for each class in validation set
        val_targets = np.array(val_dataset.targets)
        val_samples_per_class = max(num_samples // 20, 1)  # Fewer validation samples
        for class_idx in range(10):
            class_indices = np.where(val_targets == class_idx)[0]
            # Randomly select val_samples_per_class indices for this class
            if len(class_indices) > val_samples_per_class:
                selected_indices = np.random.choice(class_indices, val_samples_per_class, replace=False)
                val_indices.extend(selected_indices)
            else:
                val_indices.extend(class_indices)

        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Quick test enabled: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Set up distributed samplers if needed
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                train_dataset,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=False,
                                                seed=seed
                                                )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                                            val_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=False,
                                            drop_last=False,
                                            seed=seed
                                            )
    else:
        train_sampler = None
        val_sampler = None

    # Data loaders - for reproducible shuffling, also set a generator with fixed seed
    g = torch.Generator()
    g.manual_seed(seed)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler,
        generator=g if train_sampler is None else None
    )

    val_loader = DataLoader(
        val_dataset, batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        pin_memory=True, sampler=val_sampler
    )

    return train_loader, val_loader, train_sampler

# Setup model, optimizer, criterion and scheduler
def setup_model(model_config, device):
    """Setup model by loading the appropriate CNN version from file"""
    logger.info(f"Setting up model with config: {model_config}")

    # Determine which model version to load
    model_version = model_config.get('model_version', 1)
    model = None

    try:
        if model_version == 1:
            # Base model in main directory
            model_file = 'cnn_model'
            logger.info(f"Importing base CNN from {model_file}")
            cnn_module = importlib.import_module(model_file)
            importlib.reload(cnn_module)  # Ensure we get the latest version

        else:
            # Newer version in model_versions directory
            model_path = model_config.get('model_file', f'./model_versions/cnn_model_v{model_version}.py')
            model_path = os.path.abspath(model_path)
            logger.info(f"Attempting to load model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model_file = os.path.basename(model_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(model_file, model_path)
            cnn_module = importlib.util.module_from_spec(spec)
            sys.modules[model_file] = cnn_module
            spec.loader.exec_module(cnn_module)

        # Create model instance
        model = cnn_module.CNN()
        logger.info(f"Successfully loaded CNN model version {model_version}")

    except (ImportError, FileNotFoundError, AttributeError) as e:
        logger.error(f"Failed to load specified model version {model_version}: {e}")
        logger.info("Searching for available model versions...")

        # Look for available model versions
        available_models = []

        # Check for base model
        try:
            base_module = importlib.import_module('cnn_model')
            available_models.append((1, base_module, 'Base model'))
        except ImportError:
            pass

        # Check model_versions directory
        model_dir = './model_versions'
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith('cnn_model_v') and file.endswith('.py'):
                    try:
                        ver_num = int(file.split('_v')[1].split('.py')[0])
                        file_path = os.path.join(model_dir, file)
                        mod_name = file.replace('.py', '')

                        spec = importlib.util.spec_from_file_location(mod_name, file_path)
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[mod_name] = mod
                        spec.loader.exec_module(mod)

                        available_models.append((ver_num, mod, file_path))
                    except Exception as e:
                        logger.warning(f"Could not load {file}: {e}")

        if available_models:
            # Sort by version number (highest first)
            available_models.sort(reverse=True)
            logger.info(f"Found {len(available_models)} available model versions.")

            # Use the highest available version
            version, cnn_module, path = available_models[0]
            logger.info(f"Using model version {version} from {path}")
            model = cnn_module.CNN()
        else:
            logger.error("No usable CNN model found. Cannot continue.")
            raise RuntimeError("No CNN model implementation available")

    local_rank = model_config.get('local_rank')
    if local_rank != -1:
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP if using distributed training
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # Set up optimizer
    optimizer_type = model_config.get('optimizer', 'SGD')
    lr = model_config.get('learning_rate', 0.01)
    momentum = model_config.get('momentum', 0.9)
    weight_decay = model_config.get('weight_decay', 1e-4)

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer {optimizer_type}, using SGD")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    lr_schedule = model_config.get('lr_schedule', 'step')
    lr_steps = model_config.get('lr_steps', [30, 60, 80])
    lr_gamma = model_config.get('lr_gamma', 0.1)
    total_epochs = model_config.get('total_epochs', 90)

    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    return model, criterion, optimizer, scheduler

# Fix autograd error in the model
def fix_autograd_error(model, error_msg, traceback_info, device, args):
    """
    Generate a fixed model version that resolves autograd errors related to inplace operations
    """
    logger.error(f"Autograd error detected in model: {error_msg}")
    logger.error("Cannot automatically fix the model without LLM assistance.")
    logger.error("Try modifying your model to avoid in-place operations that interfere with gradient computation.")
    return None, None, False

# Train for one epoch
def train_epoch(epoch, model, train_loader, criterion, optimizer, device, model_config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm_epoch = 0
    layer_grad_norms_epoch = defaultdict(float)

    # Track if we've already attempted to fix an autograd error
    autograd_fix_attempted = False

    # Apply gradient clipping if configured
    if model_config.get('gradient_clip') is not None:
        clip_value = model_config.get('gradient_clip')
        logger.info(f"Using gradient clipping with value {clip_value}")

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['total_epochs']}") as t:
        batch_idx = 0
        while batch_idx < len(train_loader):
            try:
                # Get the next batch
                images, target = next(iter(train_loader))
                images, target = images.to(device), target.to(device)
                batch_idx += 1

                # Forward pass
                optimizer.zero_grad()
                output = model(images)
                if hasattr(model, 'loss_function'):
                    loss = model.loss_function(output, target)
                else:
                    loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Apply gradient clipping if configured
                if model_config.get('gradient_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.get('gradient_clip'))

                # Get gradient norms
                if model_config['local_rank'] != -1:
                    # For DDP, get gradients from the model's module
                    total_grad_norm, layer_grad_norms = get_gradient_norms(model.module)
                else:
                    total_grad_norm, layer_grad_norms = get_gradient_norms(model)

                total_grad_norm_epoch += total_grad_norm
                for layer, norm in layer_grad_norms.items():
                    layer_grad_norms_epoch[layer] += norm

                # Update weights
                optimizer.step()

                # Track statistics
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Update progress bar
                t.update(1)  # Explicitly update by 1
                t.set_postfix({
                    'loss': running_loss / batch_idx,
                    'acc': 100. * correct / total,
                })

            except RuntimeError as e:
                error_msg = str(e)
                if "inplace operation" in error_msg and not autograd_fix_attempted:
                    # This is an autograd error related to inplace operations
                    logger.error(f"Autograd error detected: {e}")
                    import traceback
                    tb_str = traceback.format_exc()

                    # Mark that we've attempted a fix
                    autograd_fix_attempted = True
                    raise RuntimeError(f"Autograd error detected: {error_msg}. Fix your model to avoid in-place operations.")
                else:
                    # Re-raise the error
                    logger.error(f"Error during training: {e}")
                    raise

    # Normalize gradient norms by number of logged iterations
    num_logged = max(1, len(train_loader) // model_config['log_interval'])
    avg_total_grad_norm = total_grad_norm_epoch / num_logged
    avg_layer_grad_norms = {layer: norm / num_logged for layer, norm in layer_grad_norms_epoch.items()}

    epoch_loss = running_loss / batch_idx  # Use actual number of batches processed
    epoch_acc = 100. * correct / max(1, total)  # Avoid division by zero

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'total_grad_norm': avg_total_grad_norm,
        'layer_grad_norms': avg_layer_grad_norms
    }

# Validate model
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total

    return {
        'loss': val_loss,
        'accuracy': val_acc
    }

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, model_config, epoch, stats, checkpoint_dir, is_final=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if is_final:
        filename = os.path.join(checkpoint_dir, f"final_model_v{model_config['model_version']}.pth")
    else:
        filename = os.path.join(checkpoint_dir, f"model_v{model_config['model_version']}_e{epoch}.pth")

    # Get model state dict (handle DDP case)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config,
        'stats': stats,
        'date': datetime.now().isoformat(),
    }

    torch.save(checkpoint, filename)
    logger.info(f"Saved checkpoint to {filename}")

    return filename

def load_model_config(filepath="model_config.json"):
    """
    Load a model configuration from a JSON file.

    Args:
        filepath (str): Path to the saved configuration file

    Returns:
        dict: The loaded model configuration or None if loading failed
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Configuration file not found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            model_config = json.load(f)

        # Convert string lists back to actual lists where needed
        for key, value in model_config.items():
            if key in ['filters', 'fc_dims', 'lr_steps'] and isinstance(value, str):
                try:
                    # Handle potential string representation of lists
                    model_config[key] = json.loads(value.replace("'", '"'))
                except:
                    # If parsing fails, leave as is
                    pass

        logger.info(f"Model configuration loaded from {filepath}")
        return model_config

    except Exception as e:
        logger.error(f"Error loading model configuration: {str(e)}")
        return None

def plot_stats(stats, model_version):
    # Set global font size
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.plot(stats['epoch_losses'], linewidth=2)
    plt.title('Training Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(stats['epoch_accuracies'], linewidth=2, label='Train')
    plt.plot(stats['val_accuracies'], linewidth=2, label='Validation')
    plt.title('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(stats['total_grad_norms'], linewidth=2)
    plt.title('Total Gradient Norm', fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Norm', fontsize=18)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # Plot gradient norms for selected layers (first, middle, last conv and fc layers)
    layer_keys = list(stats['layer_grad_norms'].keys())
    conv_layers = [k for k in layer_keys if 'conv' in k]
    fc_layers = [k for k in layer_keys if 'fc' in k or 'classifier' in k]

    selected_layers = []
    if conv_layers and len(conv_layers) >= 3:
        selected_layers.extend([conv_layers[0], conv_layers[len(conv_layers)//2], conv_layers[-1]])
    elif conv_layers:
        selected_layers.extend(conv_layers)  # Add all conv layers if fewer than 3

    if fc_layers and len(fc_layers) >= 2:
        selected_layers.extend([fc_layers[0], fc_layers[-1]])
    elif fc_layers:
        selected_layers.extend(fc_layers)  # Add all fc layers if fewer than 2

    for layer in selected_layers:
        plt.plot(stats['layer_grad_norms'][layer], label=layer, linewidth=2)

    plt.title('Layer Gradient Norms', fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Norm', fontsize=18)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Adjust legend for better readability
    if selected_layers:
        # Make layer names more readable by shortening them
        handles, labels = plt.gca().get_legend_handles_labels()
        shortened_labels = [l.split('.')[-1] if '.' in l else l for l in labels]
        plt.legend(handles, shortened_labels, fontsize=16, loc='best', frameon=True,
                  framealpha=0.7, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(f'training_stats_v{model_version}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Main function
def main(rank, world_size):
    # Set up logging
    logger.info("Starting CNN training with gradient tracking")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Parse arguments
    args = parse_args()

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")

    # Initialize distributed training if required
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl',
                               rank=rank, world_size=world_size)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create checkpoint directory
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set up data loaders
    #train_loader, val_loader, train_sampler = setup_data(args, rank, world_size, quick_test=True, num_samples=10000)
    train_loader, val_loader, train_sampler = setup_data(args, rank, world_size, quick_test=False)

    # Set up device
    device = torch.device(f"cuda:{rank}")

    # Convert args to model_config dictionary
    model_config = {
        'num_conv_layers': args.num_conv_layers,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'fc_dims': args.fc_dims,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'lr_schedule': args.lr_schedule,
        'lr_steps': args.lr_steps,
        'lr_gamma': args.lr_gamma,
        'use_skip_connections': args.use_skip_connections,
        'use_batch_norm': args.use_batch_norm,
        'gradient_clip': args.gradient_clip,
        'total_epochs': args.epochs,
        'local_rank': rank,
        'log_interval': args.log_interval,
        'model_version': 1  # Start with version 1
    }

    # LLM generated config
    if args.use_LLM_config:
        model_config = load_model_config()
    model_config['total_epochs'] = 600

    # Set up model, criterion, optimizer, scheduler
    model, criterion, optimizer, scheduler = setup_model(model_config, device)

    # Training statistics
    stats = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'total_grad_norms': [],
        'layer_grad_norms': defaultdict(list),
        'val_losses': [],
        'val_accuracies': []
    }

    # Training loop
    for epoch in range(model_config['total_epochs']):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_epoch(epoch, model, train_loader, criterion, optimizer, device, model_config)

        # Update learning rate
        scheduler.step()

        # Validate
        val_stats = validate(model, val_loader, criterion, device)

        # Update stats
        stats['epoch_losses'].append(train_stats['loss'])
        stats['epoch_accuracies'].append(train_stats['accuracy'])
        stats['total_grad_norms'].append(train_stats['total_grad_norm'])
        stats['val_losses'].append(val_stats['loss'])
        stats['val_accuracies'].append(val_stats['accuracy'])

        for layer, norm in train_stats['layer_grad_norms'].items():
            stats['layer_grad_norms'][layer].append(norm)

        # Print epoch statistics if rank is 0
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.2f}%")
            logger.info(f"Total Grad Norm: {train_stats['total_grad_norm']:.4f}")
            logger.info(f"Current LR: {optimizer.param_groups[0]['lr']}")

            # Save checkpoint periodically
            if epoch % 20 == 0 or epoch == args.epochs - 1 and epoch > 0:
                save_checkpoint(
                    model, optimizer, scheduler, model_config,
                    epoch, stats, args.checkpoint_dir,
                    is_final=(epoch == args.epochs - 1)
                )

            # Generate plots periodically
            if epoch % 2 == 0:
                plot_stats(stats, model_config['model_version'])

    # Save the final model (rank 0 only)
    if rank == 0:
        # Save final checkpoint
        save_checkpoint(model, optimizer, scheduler, model_config, args.epochs-1,
                      stats, args.checkpoint_dir, is_final=True)

        # Plot training statistics
        plot_stats(stats, model_config['model_version'])

        logger.info("Training complete! Final model saved.")


#%%
if __name__ == "__main__":

    import torch.multiprocessing as mp
    world_size = 4
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)