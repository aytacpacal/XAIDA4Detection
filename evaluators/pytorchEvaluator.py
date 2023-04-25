#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from utils import *

from .visualizers import *

class PytorchEvaluator():
    """
    Evaluator generic class
    """
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.loader = dataloader
    
    def evaluate(self, inference_outputs):
        """
        Evaluate model
        """
        if self.config['evaluation']['visualization']:
            visualizer = globals()['ClassificationVisualizer'+str(self.config['arch']['output_model_dim'])+'D'](self.config, self.model, self.loader)
            visualizer.visualize(inference_outputs)

        return inference_outputs
    
    def inference(self):
        """
        Infere results
        """
        test_outputs = []
        test_labels = []
        test_time = []
        test_event_names = []

        self.model.eval()
        with torch.no_grad():
            for index, sample in enumerate((pbar := tqdm(self.loader))):
                pbar.set_description('Infering Dataloader')
                if not 'masks' in sample.keys():
                    sample['masks'] = torch.ones(sample['x'].shape)
                if not 'time' in sample.keys():
                    sample['time'] = index
                if not 'event_name' in sample.keys():
                    sample['event_name'] = index

                x, masks, labels = adapt_variables(self.config, sample['x'], sample['masks'], sample['labels'])
                output = self.model(x)

                if isinstance(output, tuple):
                    output = output[-1]

                output, labels, time, event_name = self.adapt_input(output, labels, sample['time'], sample['event_name'])
                
                test_outputs.append(output)
                test_labels.append(labels)
                test_time.append(time)
                test_event_names.append(event_name)
                        
            return {'outputs':test_outputs, 'labels':test_labels, 'time':test_time, 'event_names': test_event_names}
    
    def adapt_input(self, output, labels, time, event_name):
        """
        Adapt input size for evaluation purposes
        """
        if self.config['arch']['output_model_dim'] == 1:
            output = output[0]
            labels = labels[0]
        
        elif self.config['arch']['output_model_dim'] == 2:
            output = output[0]
            labels = labels[0,0]

        elif self.config['arch']['output_model_dim'] == 3:
            output = output[0,:,0]
            labels = labels[0,0,0]
        
        if not isinstance(time, int):
            time = time[0]

        if not isinstance(event_name, int):
            event_name = event_name[0]

        return output, labels, time, event_name

        
        