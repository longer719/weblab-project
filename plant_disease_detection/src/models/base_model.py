# src/models/base_model.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])