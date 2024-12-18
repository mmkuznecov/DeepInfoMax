import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseclasses import (
    BaseEncoder, 
    BaseDecoder,
    BaseGlobalDiscriminator,
    BaseLocalDiscriminator,
    BasePriorDiscriminator
)

class CIFAR10Encoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512*20*20, 64)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)
        
        self.features_shape = (128, 26, 26)  # Shape after c1

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features

class CIFAR10Decoder(BaseDecoder):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 512*20*20)
        
        self.dc3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1)
        self.dc2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1)
        self.dc1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1)
        self.dc0 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=1)
        
        self.b3 = nn.BatchNorm2d(256)
        self.b2 = nn.BatchNorm2d(128)
        self.b1 = nn.BatchNorm2d(64)

    def forward(self, encoded, features=None):
        h = self.l1(encoded)
        h = h.view(h.size(0), 512, 20, 20)
        
        h = F.relu(self.b3(self.dc3(h)))
        h = F.relu(self.b2(self.dc2(h)))
        h = F.relu(self.b1(self.dc1(h)))
        reconstructed = torch.sigmoid(self.dc0(h))
        return reconstructed

class CIFAR10GlobalDiscriminator(BaseGlobalDiscriminator):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class CIFAR10LocalDiscriminator(BaseLocalDiscriminator):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)

class CIFAR10PriorDiscriminator(BasePriorDiscriminator):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))