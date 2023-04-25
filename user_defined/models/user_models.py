#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class UserDefined_LSTM(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        # Input dimensions
        input_dim = np.prod(config['data']['input_size'])

        # Number of classes
        if config['data']['num_classes'] > 2:
            classes = config['data']['num_classes']
        else:
            classes = 1
        
        # Batch size
        self.batch_size = config['implementation']['trainer']['batch_size']
        
        # Hidden dimensions
        self.hidden_dim = 1#config['arch']['args']['hidden_dim']
        
        # Number of hidden layers
        self.n_layer = config['arch']['params']['n_layers']
        
        # LSTM definition
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layer, batch_first= True, dropout=config['arch']['params']['dropout'])
        
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass
        """
        out_lstm, _ = self.lstm(x)
        out_fc = self.fc(out_lstm[:, -1, :])

        return out_fc
