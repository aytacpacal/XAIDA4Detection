#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from utils import *

class ClassificationVisualizerAbstract(ABC):
    """
    Evaluator generic class
    """
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.test_loader = dataloader
    
    def visualize(self, inference_outputs):
        """
        Visualize results
        """
        output = inference_outputs['outputs']
        labels = inference_outputs['labels']
        time = inference_outputs['time']
        event_names= inference_outputs['event_names']
        
        self.per_sample_operations(output, labels, time, event_names)
        self.global_operations()
    
    @abstractmethod
    def per_sample_operations(self):
        pass
    
    @abstractmethod
    def global_operations(self):
        pass