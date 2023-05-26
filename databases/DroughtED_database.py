#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import torch
import torch.nn.functional as F
import dask
dask.config.set(scheduler='synchronous')

class DroughtED(torch.utils.data.Dataset):
    """
    DroughtED Database
    """
    def __init__(self, config, period = 'train'):
        self.config = config['data'] # configuration file
        self.period = period # train/val/test
        self.window_size = self.config['window_size']
        self.features_selected = self.config['features_selected']
        self.num_classes = self.config['num_classes']
        self.train_slice = self.config['train_slice']
        self.val_slice = self.config['val_slice']
        self.test_slice = self.config['test_slice']
        
        # Class IDs
        self.id2class = {
            'None': 0,
            'D0': 1,
            'D1': 2,
            'D2': 3,
            'D3': 4,
            'D4': 5,
        }

        self.scaler_dict = {}
        self.scaler_dict_past = {}

        # Read database filenames
        self.dfs = self.read_database_files()
        if self.num_classes == 2:
            self.binarize_data() 
       
        # Data pre-processing
        data = self.loadXY(period = self.period)
        self.X, self.y = data[0], data[1]
   
        ones_percentage = np.count_nonzero(self.y)*100/len(self.X) 
        self.weights = [100-ones_percentage, ones_percentage]
        print(self.weights)
    
    def read_database_files(self):
        files = {}

        for dirname, _, filenames in os.walk(os.path.join(self.config['root'], self.config['data_file'])):
            for filename in filenames:
                if 'val' in filename: 
                    files['val'] = os.path.join(dirname, filename)
                if 'test' in filename: 
                    files['test'] = os.path.join(dirname, filename)

        df_list = []
        for split in ['val', 'test']:
            df = pd.read_csv(files[split]).set_index(['fips', 'date'])
            df_list.append(df)

        dfs = pd.concat(df_list, axis=0, ignore_index=False)
        self.dfs = self.select_slice(dfs)
        return self.dfs 

    def select_slice(self, dfs):
        dates = dfs.index.get_level_values(1)
        slice = getattr(self, self.period+'_slice')
        start = slice['start']
        end = slice['end']
        dfs = dfs[(dates > start) & (dates < end)].copy()
        return dfs

    def binarize_data(self):
        class_bound = self.config['class_bound']
        self.dfs.loc[self.dfs["score"] < class_bound] = 0
        self.dfs.loc[self.dfs["score"] >= class_bound] = 1
                                    
    def loadXY(self,
        period='train', # data period to load
        random_state=42, # keep this at 42
        normalize=True, # standardize data
    ):
        """
        Load database
        """

        ## Initialize random state
        if random_state is not None:
            np.random.seed(random_state)

        ## Get column's name for the meteorological indicatos
        time_data_cols = sorted([c for c in self.dfs.columns if c not in ["fips", "date", "score"]])
        time_data_cols = [time_data_cols[int(i)] for i in self.features_selected]

        ## Filter all data point that do not have a score defined (NaN value)
        score_df = self.dfs.dropna(subset=["score"])

        ## Create variables to store the data
        max_buffer = 7
        X_time = np.empty((len(self.dfs) // self.window_size , self.window_size, len(time_data_cols)))
        y_target = np.empty((len(self.dfs) // self.window_size , 1))
        
        count = 0
        ## Iteration over the fips
        # print(self.dfs.index.get_level_values(0))
        for fips in tqdm(score_df.index.get_level_values(0).unique()):
        # for fips in tqdm(score_df.index.get_level_values(0).unique().values[:100]):

            ## Select randomly where to start sampling
            if random_state is not None and self.window_size != 1:
                start_i = np.random.randint(1, self.window_size)
            else:
                start_i = 1
            
            ## Get all samples with the fips ID that we are evaluating 
            fips_df = self.dfs[(self.dfs.index.get_level_values(0) == fips)]
            X = fips_df[time_data_cols].values
            y = fips_df["score"].values

            for idx, i in enumerate(range(start_i, len(y) - (self.window_size + max_buffer), self.window_size)):
                ## Save window of samples
                X_time[count, :, : len(time_data_cols)] = X[i : i + self.window_size]
                
                ## 
                temp_y = y[i + self.window_size : i + self.window_size + max_buffer]
                y_target[count] = int(np.around(np.array(temp_y[~np.isnan(temp_y)][0])))

                count += 1

        print(f"loaded {count} samples")

        # Normalize the data
        if normalize:
            X_time = self.normalize(X_time)
        data = [X_time[:count], y_target[:count]]


        return tuple(data)
    
    def interpolate_nans(self, padata, pkind='linear'):
        """
        see: https://stackoverflow.com/a/53050216/2167159
        """
        aindexes = np.arange(padata.shape[0])
        agood_indexes, = np.where(np.isfinite(padata))
        f = interp1d(agood_indexes
               , padata[agood_indexes]
               , bounds_error=False
               , copy=False
               , fill_value="extrapolate"
               , kind=pkind)
        return f(aindexes)
                                                                                
            
    def normalize(self, X_time):
        """
        Get statistics for standardization
        """
        X_time_train = self.loadXY(period = 'train', normalize=False)
        
        X_time_train = X_time_train[0]
        for index in tqdm(range(X_time.shape[-1])):
            # Fit data    
            # self.scaler_dict[index] = RobustScaler().fit(X_time_train[:, :, index].reshape(-1, 1))
            self.scaler_dict[index] = StandardScaler().fit(X_time_train[:, :, index].reshape(-1, 1))
            X_time[:, :, index] = (
                self.scaler_dict[index]
                .transform(X_time[:, :, index].reshape(-1, 1))
                .reshape(-1, X_time.shape[-2])
            )
        X_time = np.clip(X_time, a_min=-3., a_max=3.) / 3.
        index = 0
        return X_time                                          
                                                    
    def __getitem__(self, index):
        """
        Get item from the dataset
        """

        if self.num_classes == 2:
            return {'x': torch.Tensor(self.X[index, :]) , 'labels': torch.Tensor(self.y[index])}
        else:
            return {'x': torch.Tensor(self.X[index, :]), 
                'labels': F.one_hot(torch.Tensor(self.y[index]).squeeze().long(),num_classes=self.num_classes)}

    def __getallitems__(self):
        """
        Get the whole dataset
        """
        return {'x': np.transpose(np.squeeze(self.X), (1,0)), 'labels': np.squeeze(self.y)}
        
    def __len__(self):
        """
        Returns:
            (int): the length of the dataset (number of samples)
        """
        return len(self.X)