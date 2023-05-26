#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class UD_LSTM(nn.Module):
    """
    LSTM for extreme event detection
    """
    def __init__(self, config):
        super().__init__()
        
        input_dim = np.prod(config['data']['input_size'])

        if config['data']['num_classes'] > 2:
            classes = config['data']['num_classes']
        else:
            classes = 1
        
        
        # Hidden dimensions
        self.hidden_dim = config['arch']['args']['hidden_dim']
        
        # Number of hidden layers
        self.n_layer = 1
        
        # LSTM definition
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layer, batch_first = True)
        
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, classes)
        # self.fc = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        # self.fc2 = nn.Linear(int(self.hidden_dim/2), classes)
    
    def forward(self, x):
        """
        Forward pass
        """

        # Initialize hidden state with zeros
        #h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_dim).requires_grad_()
    
        # Initialize cell state
        #c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_dim).requires_grad_()
    
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = self.lstm(x)
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        # out = self.fc2(F.relu(self.fc(out[:, -1, :])))
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
