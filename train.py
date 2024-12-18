import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from pathlib import Path
from datetime import datetime
import argparse

from datasets.cifar10_dataset import CIFAR10Data
from models.cifar10 import (
    CIFAR10Encoder,
    CIFAR10Decoder,
    CIFAR10GlobalDiscriminator,
    CIFAR10LocalDiscriminator,
    CIFAR10PriorDiscriminator
)
from training.losses import DeepInfoMaxLoss, ReconstructionLoss
from datasets.tensor_storage import TensorStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_deepinfomax(args):
    """Train using Deep InfoMax approach."""
    # Initialize models
    encoder = CIFAR10Encoder().to(args.device)
    global_disc = CIFAR10GlobalDiscriminator().to(args.device)
    local_disc = CIFAR10LocalDiscriminator().to(args.device)
    prior_disc = CIFAR10PriorDiscriminator().to(args.device)
    
    # Initialize loss and optimizers
    criterion = DeepInfoMaxLoss(
        global_disc=global_disc,
        local_disc=local_disc,
        prior_disc=prior_disc,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    ).to(args.device)
    
    encoder_optim = Adam(encoder.parameters(), lr=args.learning_rate)
    disc_optim = Adam(criterion.parameters(), lr=args.learning_rate)
    
    return train_loop_deepinfomax(
        encoder=encoder,
        criterion=criterion,
        encoder_optim=encoder_optim,
        disc_optim=disc_optim,
        dataloaders=args.dataloaders,
        args=args
    )

def train_autoencoder(args):
    """Train using autoencoder approach."""
    # Initialize models
    encoder = CIFAR10Encoder().to(args.device)
    decoder = CIFAR10Decoder().to(args.device)
    
    # Initialize loss and optimizer
    criterion = ReconstructionLoss().to(args.device)
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate
    )
    
    return train_loop_autoencoder(
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=args.dataloaders,
        args=args
    )

def train_loop_deepinfomax(encoder, criterion, encoder_optim, disc_optim, dataloaders, args):
    """Training loop for Deep InfoMax approach."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        encoder.train()
        train_loss = 0
        batch = tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for x, _ in batch:
            x = x.to(args.device)
            
            encoder_optim.zero_grad()
            disc_optim.zero_grad()
            
            # Forward pass
            y, M = encoder(x)
            # Create M_prime by rotating the batch
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            
            # Compute loss
            loss = criterion(y, M, M_prime)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            encoder_optim.step()
            disc_optim.step()
            
            batch.set_postfix({'loss': loss.item()})
            
        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        
        # Validation
        encoder.eval()
        val_loss = 0
        with torch.no_grad():
            batch = tqdm(dataloaders['val'], desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for x, _ in batch:
                x = x.to(args.device)
                y, M = encoder(x)
                M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                loss = criterion(y, M, M_prime)
                val_loss += loss.item()
                batch.set_postfix({'loss': loss.item()})
                
        val_loss /= len(dataloaders['val'])
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(encoder, None, 'best_encoder.pth', args.exp_dir)
            
        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return encoder, train_losses, val_losses

def train_loop_autoencoder(encoder, decoder, criterion, optimizer, dataloaders, args):
    """Training loop for autoencoder approach."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        encoder.train()
        decoder.train()
        train_loss = 0
        batch = tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for x, _ in batch:
            x = x.to(args.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            encoded, features = encoder(x)
            reconstructed = decoder(encoded, features)
            
            # Compute loss
            loss = criterion(reconstructed, x)
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch.set_postfix({'loss': loss.item()})
            
        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        
        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            batch = tqdm(dataloaders['val'], desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for x, _ in batch:
                x = x.to(args.device)
                encoded, features = encoder(x)
                reconstructed = decoder(encoded, features)
                loss = criterion(reconstructed, x)
                val_loss += loss.item()
                batch.set_postfix({'loss': loss.item()})
                
        val_loss /= len(dataloaders['val'])
        val_losses.append(val_loss)
        
        # Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(encoder, decoder, 'best_models.pth', args.exp_dir)
            
        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return encoder, decoder, train_losses, val_losses

def save_model(encoder, decoder, filename, save_dir):
    """Save model(s) to disk."""
    save_dict = {
        'encoder_state_dict': encoder.state_dict(),
    }
    if decoder is not None:
        save_dict['decoder_state_dict'] = decoder.state_dict()
    
    save_path = os.path.join(save_dir, filename)
    torch.save(save_dict, save_path)
    logger.info(f'Saved model to {save_path}')

def plot_losses(train_losses, val_losses, save_dir):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

def save_embeddings(encoder, dataloaders, exp_name, args):
    """Save embeddings using TensorStorage."""
    encoder.eval()
    embeddings = []
    metadata = []
    
    with torch.no_grad():
        # Process train set
        for i, (x, y) in enumerate(tqdm(dataloaders['train'], desc='Processing train set')):
            x = x.to(args.device)
            encoded, _ = encoder(x)
            embeddings.extend(encoded.cpu().numpy())
            metadata.extend([{
                'split': 'train',
                'label': label.item(),
                'batch_idx': i
            } for label in y])
            
        # Process test set
        for i, (x, y) in enumerate(tqdm(dataloaders['test'], desc='Processing test set')):
            x = x.to(args.device)
            encoded, _ = encoder(x)
            embeddings.extend(encoded.cpu().numpy())
            metadata.extend([{
                'split': 'test',
                'label': label.item(),
                'batch_idx': i
            } for label in y])
    
    # Create storage
    storage_dir = os.path.join('storages', exp_name)
    description = f"CIFAR-10 embeddings from {exp_name} experiment"
    
    storage = TensorStorage.create_storage(
        storage_dir=storage_dir,
        data_iterator=iter(embeddings),
        metadata_iterator=iter(metadata),
        description=description
    )
    
    logger.info(f'Saved embeddings to {storage_dir}')
    return storage

def main():
    parser = argparse.ArgumentParser(description='Train Deep InfoMax or Autoencoder on CIFAR-10')
    
    # Training settings
    parser.add_argument('--mode', type=str, choices=['deepinfomax', 'autoencoder'], required=True,
                        help='Training mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # Deep InfoMax specific
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha weight in Deep InfoMax loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta weight in Deep InfoMax loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma weight in Deep InfoMax loss')
    
    # Other settings
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set up experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"{args.mode}_{timestamp}"
    args.exp_dir = os.path.join('results', args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # Save experiment config
    with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Prepare data
    data = CIFAR10Data(batch_size=args.batch_size)
    data.prepare_data()
    data.setup()
    args.dataloaders = data.get_dataloaders()
    
    # Train model
    logger.info(f'Starting {args.mode} training...')
    if args.mode == 'deepinfomax':
        encoder, train_losses, val_losses = train_deepinfomax(args)
    else:  # autoencoder
        encoder, decoder, train_losses, val_losses = train_autoencoder(args)
    
    # Plot and save training curves
    plot_losses(train_losses, val_losses, args.exp_dir)
    
    # Save embeddings
    storage = save_embeddings(encoder, args.dataloaders, args.exp_name, args)
    
    logger.info('Training completed successfully!')

if __name__ == '__main__':
    main()