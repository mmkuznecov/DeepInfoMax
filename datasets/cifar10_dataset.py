import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from typing import Tuple, Dict

class CIFAR10Data:
    """CIFAR10 data module for managing dataset and dataloaders."""
    
    def __init__(
        self,
        data_dir: str = 'data',
        batch_size: int = 128,
        train_val_split: float = 0.9,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize CIFAR10 data module.
        
        Args:
            data_dir: Directory to store dataset
            batch_size: Batch size for training
            train_val_split: Fraction of training data to use for training (rest for validation)
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def prepare_data(self):
        """Download data if needed."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        
    def setup(self) -> None:
        """Setup train, validation and test datasets."""
        # Load datasets
        trainval_dataset = CIFAR10(
            self.data_dir,
            train=True,
            transform=ToTensor()
        )
        self.test_dataset = CIFAR10(
            self.data_dir,
            train=False,
            transform=ToTensor()
        )
        
        # Split training and validation
        train_length = int(len(trainval_dataset) * self.train_val_split)
        val_length = len(trainval_dataset) - train_length
        
        self.train_dataset, self.val_dataset = random_split(
            trainval_dataset, 
            [train_length, val_length],
            generator=torch.Generator().manual_seed(42)
        )
        
    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get dictionary containing all dataloaders.
        
        Returns:
            Dict with train, val, and test dataloaders
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def get_dims(self) -> Tuple[int, int, int]:
        """Get input dimensions of dataset (channels, height, width)."""
        return 3, 32, 32  # CIFAR10 dimensions