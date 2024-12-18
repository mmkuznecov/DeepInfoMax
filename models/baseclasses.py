import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseEncoder(nn.Module, ABC):
    """Base encoder class that all encoders should inherit from."""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        """Forward pass returning encoded representation and features"""
        pass
    
    def get_features_shape(self):
        """Return shape of intermediate features"""
        return self.features_shape

class BaseDecoder(nn.Module, ABC):
    """Base decoder class that all decoders should inherit from."""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, encoded, features=None):
        """Forward pass reconstructing input from encoded representation"""
        pass

class BaseGlobalDiscriminator(nn.Module, ABC):
    """Base global discriminator for Deep InfoMax."""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, y, M):
        """Score global feature-representation pairs."""
        pass

class BaseLocalDiscriminator(nn.Module, ABC):
    """Base local discriminator for Deep InfoMax."""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        """Score local feature-representation pairs."""
        pass

class BasePriorDiscriminator(nn.Module, ABC):
    """Base prior discriminator for Deep InfoMax."""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        """Discriminate between samples from prior and encoded representations."""
        pass