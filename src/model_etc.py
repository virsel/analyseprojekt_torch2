import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import numpy as np
from joblib import dump, load
from glob import glob
import json
import os
import torch
from logger import TensorLogger
import torch.optim as optim
from datetime import datetime
from config import Config
    
class EmbBlock(nn.Module):
    def __init__(self, 
                 in_dim=768,
                 dropout_rate=0.3):
        super(EmbBlock, self).__init__()
        
        # Gradual dimension reduction
        self.fc1 = nn.Linear(in_dim, 512)
        # Gradual dimension reduction
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Add projection layers for residual connections
        self.proj2 = nn.Linear(512, 128)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_etc):
        # Main path
        x = self.fc1(X_etc)
        x = self.bn1(x)
        x = F.relu(x)
        x1 = self.dropout(x)
        
        x = self.fc2(x1)
        x = self.bn2(x)
        x = F.relu(x)
        x2 = self.dropout(x)
        
        x = self.fc3(x2)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Add residual from previous layer
        x = x + self.proj2(x1)
        
        x = self.dropout(x)
        
        return x

class EtcBlock(nn.Module):
    def __init__(self, 
                 in_dim=768,
                 dropout_rate=0.3):
        super(EtcBlock, self).__init__()
        
        # Gradual dimension reduction
        self.fc1 = nn.Linear(in_dim, 512)
        # Gradual dimension reduction
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Add projection layers for residual connections
        self.proj2 = nn.Linear(512, 128)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_etc):
        # Main path
        x = self.fc1(X_etc)
        x = self.bn1(x)
        x = F.relu(x)
        x1 = self.dropout(x)
        
        x = self.fc2(x1)
        x = self.bn2(x)
        x = F.relu(x)
        x2 = self.dropout(x)
        
        x = self.fc3(x2)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Add residual from previous layer
        x = x + self.proj2(x1)
        
        x = self.dropout(x)
        
        return x