"""
Training Script for Interactive Segmentation Models

Usage:
    python tools/train.py --config configs/gaussian_s3.yaml
    python tools/train.py --config configs/binary.yaml
    python tools/train.py --config configs/disk_r3.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lits_dataset import LiTSInteractiveDataset
from models.unet3d_interactive import UNet3DInteractive
from utils.losses import DiceLoss, CombinedLoss
from utils.metrics import compute_dice_score


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config):
    """Setup experiment directory and save config."""
    exp_name = config['experiment']['name']
    output_dir = os.path.join(config['experiment']['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")

    return output_dir


def create_datasets(config):
    """Create train and validation datasets."""
    data_config = config['data']

    # Training dataset
    train_dataset = LiTSInteractiveDataset(
        data_root=data_config['data_root'],
        split='train',
        encoder_type=data_config['encoder']['type'],
        encoder_kwargs={k: v for k, v in data_config['encoder'].items() if k != 'type'},
        simulator_kwargs=data_config['simulator'],
        patch_size=tuple(data_config['patch_size']),
        num_classes=data_config['num_classes']
    )

    # Validation dataset
    val_dataset = LiTSInteractiveDataset(
        data_root=data_config['data_root'],
        split='val',
        encoder_type=data_config['encoder']['type'],
        encoder_kwargs={k: v for k, v in data_config['encoder'].items() if k != 'type'},
        simulator_kwargs=data_config['simulator'],
        patch_size=tuple(data_config['patch_size']),
        num_classes=data_config['num_classes']
    )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create data loaders."""
    data_config = config['data']

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(config, device):
    """Create model and move to device."""
    model_config = config['model']
    model_type = model_config.get('type', 'unet3d_interactive')

    if model_type == 'fastsam3d_interactive':
        from models import FastSAM3DInteractive
        model = FastSAM3DInteractive(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            image_size=tuple(model_config['image_size']),
            embed_dim=model_config.get('embed_dim', 192),
            depth=model_config.get('depth', 6),
            num_heads=model_config.get('num_heads', 6),
            use_pretrained=model_config.get('use_pretrained', False),
            pretrained_path=model_config.get('pretrained_path', None)
        )
    else:
        # Default: U-Net
        from models import UNet3DInteractive
        model = UNet3DInteractive(
            in_channels=model_config['in_channels'],
            prompt_channels=model_config['prompt_channels'],
            out_channels=model_config['out_channels'],
            base_filters=model_config['base_filters'],
            depth=model_config['depth']
        )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    return model


def create_optimizer_scheduler(model, config, steps_per_epoch):
    """Create optimizer and learning rate scheduler."""
    train_config = config['training']

    # Optimizer
    if train_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=0.9,
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")

    # Scheduler
    if train_config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['max_epochs'] * steps_per_epoch
        )
    elif train_config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50 * steps_per_epoch,
            gamma=0.5
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    train_config = config['training']

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        image = batch['image'].to(device)
        prompts = batch['prompts'].to(device)
        mask = batch['mask'].to(device)

        # Prompt dropout (train without prompts sometimes)
        if np.random.rand() < train_config.get('prompt_dropout', 0.0):
            prompts = torch.zeros_like(prompts)

        # Forward pass
        optimizer.zero_grad()
        logits = model(image, prompts)
        # debug
        print("logits:", logits.shape)
        print("mask:", mask.shape)

        loss = criterion(logits, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Compute metrics
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            dice = compute_dice_score(pred, mask, num_classes=config['data']['num_classes'])

        # Update statistics
        total_loss += loss.item()
        total_dice += dice
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches

    return avg_loss, avg_dice


def validate(model, val_loader, criterion, device, config):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            image = batch['image'].to(device)
            prompts = batch['prompts'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            logits = model(image, prompts)
            loss = criterion(logits, mask)

            # Compute metrics
            pred = torch.argmax(logits, dim=1)
            dice = compute_dice_score(pred, mask, num_classes=config['data']['num_classes'])

            # Update statistics
            total_loss += loss.item()
            total_dice += dice
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches

    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train interactive segmentation model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup experiment
    output_dir = setup_experiment(config)

    # Set random seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create datasets and loaders
    print("\nCreating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(
        model, config, len(train_loader)
    )

    # Create loss function
    criterion = CombinedLoss(
        dice_weight=config['training']['loss']['dice_weight'],
        ce_weight=config['training']['loss']['ce_weight'],
        num_classes=config['data']['num_classes']
    )

    # Training loop
    print("\nStarting training...")
    best_dice = 0.0

    for epoch in range(1, config['training']['max_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['max_epochs']}")
        print('='*60)

        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, epoch
        )

        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")

        # Validate
        if epoch % config['training']['val_interval'] == 0:
            val_loss, val_dice = validate(model, val_loader, criterion, device, config)
            print(f"Val - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                checkpoint_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'config': config
                }, checkpoint_path)
                print(f"Saved best model (Dice: {best_dice:.4f})")

        # Save checkpoint periodically
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)

    print(f"\nTraining complete! Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()