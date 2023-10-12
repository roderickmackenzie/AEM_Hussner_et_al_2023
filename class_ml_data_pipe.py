#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   AI JV-curve extraction tool
#   Copyright (C) 2023 Markus Hußner and Roderick C. I. MacKenzie
#	roderick dot mackenzie at durham ac uk
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

import os
import numpy as np
import pandas as pd
from scipy import interpolate

import logging
import logging.config
logger_path = os.path.join(os.path.expanduser("~"), 'ML_code', 'logger')
logging.config.fileConfig(fname=os.path.join(logger_path, 'file.conf'), disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger('fileLogger')
logger.setLevel('INFO')

class machinelearning_data_struct(object):
    """
    this class handles the feature and target splitting and test and training set splitting and the normalisation of the data
    """
    def __init__(self, data, dic_features, dic_targets, target_normalization_axis='col', target_normalization_splitting = ['mu', 'tau'], feature_normalization_axis='block', feature_normalization_splitting = ['dark', 'light'], noise=False):
        #full dataset
        self.dataset_name = ''
        self.data = data
        self.data_names = self.data.columns
        # if noise should be added to the data
        self.noise = noise
        if self.noise == True:
            self.data = self.create_noise(self.data)
        # features
        # how to normalize the feature vectors: normalize the dataframe columnwise, rowwise or blockwise
        self.feature_normalization_axis = feature_normalization_axis
        # if normalization is blockwise, what are the blocks called, list of strings 
        self.feature_normalization_splitting = feature_normalization_splitting
        # dictionary that contains the features and if they have to be logged before normalisation
        self.dic_features = dic_features
        self.features = self.select_data(dic_features)
        self.features_normalized, self.normalization_features = self.normalize_data(self.features, self.dic_features, axis=self.feature_normalization_axis, regex=self.feature_normalization_splitting)
        #targets
        self.target_normalization_axis = target_normalization_axis
        self.target_normalization_splitting = target_normalization_splitting
        self.dic_targets = dic_targets
        self.targets = self.select_data(dic_targets)
        self.targets_normalized, self.normalization_targets = self.normalize_data(self.targets, self.dic_targets, axis=self.target_normalization_axis, regex=self.target_normalization_splitting)
        
        #train, test split
        self.features_training, self.features_test  = self.split_test_train_data(self.features_normalized, split_fraction=0.8)
        self.targets_training, self.targets_test = self.split_test_train_data(self.targets_normalized, split_fraction=0.8)
        
        if self.feature_normalization_axis == 'block':
            self.denormed_features_test = self.denormalize_min_max_blockwise(self.features_test, self.normalization_features, self.feature_normalization_splitting)
            self.denormed_features_training = self.denormalize_min_max_blockwise(self.features_training, self.normalization_features, self.feature_normalization_splitting)
            
    def split_test_train_data(self, data, split_fraction = 0.8):
        """
        split the data in training and test set by split_fraction
        """
        train_data = data.sample(frac=split_fraction, random_state=1)
        test_data = data.drop(train_data.index)
        return train_data, test_data
    
    def select_data(self, dic):
        """
        selects data from a pandas dataframe based on the coulmn names passed in dict

        Parameters
        ----------
        dic : dictionary
            contains the names of the columns to select from data

        Returns
        -------
        data : pandas dataframe
            dataframe only containing the selected columns

        """
        data_dic = {}
        for col in dic.keys():
            data_dic[col] = self.data[col]
        data = pd.DataFrame(data_dic)
        return data
    
    def normalize_data(self, data, dic, axis = 'col', regex = []):
        """
        normalize the data

        Parameters
        ----------
        data : pandas dataframe
            data to be normalised
        dic : dictionary
            dictionary containing the columns to normalise and wether or not they have to be log
        axis : string, optional
            'col' for clumn-wise, 'row' for row-wise, 'block' for block-wise. The default is 'col'.
        regex : list of strings, optional
            regular expression to be used to split data into blocks to normalise. The default is [].

        Returns
        -------
        data_normalized : pandas dataframe
            normalised data
        normalization : normalisation object
            e.g. norm_object_min_max containing info to use same normalisation on other data

        """
        #logger.info(data)
        #logger.info(dic)
        
        # take log of data, where indicated in dic
        data = self.log_scale(data, dic)
        if axis == 'row':
            data_normalized, normalization = self.normalize_minmax_rowwise(data)
            #data_normalized, normalization = self.normalize_standard_rowwise(data)
        elif axis == 'col':
            data_normalized, normalization = self.normalize_minmax_columnwise(data)
            #data_normalized, normalization = self.normalize_standard_columnwise(data)
        elif axis == 'block':
            data_normalized, normalization = self.normalize_minmax_blockwise(data, regex, axis = 0)
            #data_normalized, normalization = self.normalize_standard_blockwise(data, regex)
        data_normalized = self.create_df_from_processed_data(data_normalized, data.columns)
        return data_normalized, normalization
    
    def normalize_minmax_columnwise(self, df):
        """
        normalise the data column-wise axis = 0
        """
        norm = norm_object_min_max(df, axis=0)
        df_norm = norm.min_max_norm(df)
        return df_norm, norm
    
    def normalize_minmax_rowwise(self, df):
        """
        normalise the data row-wise axis = 1
        """
        norm = norm_object_min_max(df, axis=1)
        df_norm = norm.min_max_norm(df)
        return df_norm, norm
    
    def normalize_minmax_blockwise(self, data, regex, axis = 0):
        """
        normalise the data block-wise
        regex is a list with strings/sub-strings defining the blocks
        then each block gets normalised (column-wise)
        
        create dictionary with normalisation object for each regex for later use
        """
        norm = {}
        normed_data = {}
        for exp in regex:
            data_block = data.filter(regex=exp)
            normalize = norm_object_min_max(data_block, axis = axis)
            norm[exp] = normalize
            normed_data[exp] = norm[exp].min_max_norm(data_block)
        
        data_rows = {}
        for norm_block_df in normed_data.values():
            for col in norm_block_df.columns:
                data_rows[col] = norm_block_df[col]
        
        df_norm = pd.DataFrame(data_rows)
        return df_norm, norm
    
    def denormalize_min_max_blockwise(self, df_norm, norm, regex, axis=0):
        denormed_data = {}
        for exp in regex:
            data_block = df_norm.filter(regex=exp)
            denormed_data[exp] = norm[exp].min_max_denorm(data_block)
        
        data_rows = {}
        for denorm_block_df in denormed_data.values():
            for col in denorm_block_df.columns:
                data_rows[col] = denorm_block_df[col]
        
        df_denorm = pd.DataFrame(data_rows)
        return df_denorm
    
    # same procedure but instead of min-max with standard normalization
    def normalize_standard_columnwise(self, df):
        norm = norm_object_standard(df)
        df_norm = norm.standard_norm(df)
        return df_norm, norm
    
    def normalize_standard_rowwise(self, df):
        norm = norm_object_standard(df)
        df_norm = norm.standard_norm(df)
        return df_norm, norm
    
    def normalize_standard_blockwise(self, data, regex):
        norm = {}
        normed_data = {}
        for exp in regex:
            data_block = data.filter(regex=exp)
            normalize = norm_object_standard(data_block)
            norm[exp] = normalize
            normed_data[exp] = norm[exp].standard_norm(data_block)
        
        data_rows = {}
        for norm_block_df in normed_data.values():
            for col in norm_block_df.columns:
                data_rows[col] = norm_block_df[col]
        
        df_norm = pd.DataFrame(data_rows)
        return df_norm, norm 
    
    def log_scale(self, data, dic):
        """
        take log of data if indicated in dic
        """
        for col in data.columns:
                if dic[col]['log']=='y':
                    data[col] = np.log10(np.abs(data[col]))
        return data
        
    def create_noise(self, df):
        """
        return a dataset with added white noise
        percentage gives the magnitude of the noise in relation to the original data
        """
        percentage = 0.05
        df_rows = []
        for row in df.iterrows():
            row_dict = row[1].to_dict()
            
            for split in self.feature_normalization_splitting:
                jv_row = row[1].filter(regex=f'{split}')
                temp_x = np.arange(len(jv_row))
                f = interpolate.interp1d(temp_x, jv_row)
                x = np.linspace(0, temp_x[-1], 1000)
                y = f(x)
                y_std = np.std(y)
                y_noise = np.random.normal(0,y_std,1000)
                y = f(x) + y_noise*percentage
                f_noise = interpolate.interp1d(x, y)
                for i,name in enumerate(jv_row.keys()):
                    row_dict[name] = f_noise(temp_x[i])
            df_rows.append(row_dict)
        df_noise = pd.DataFrame(df_rows)
        
        df.update(df_noise)
        logger.info('-------------------------- noise applied --------------------------')
        return df
    
    def create_df_from_processed_data(self, data_array, column_names):
        """
        create a pandas dataframe from array giving it column_names as column names
        """
        df = pd.DataFrame(data = data_array, columns=column_names)
        return df
       
class norm_object_min_max():
    """
    class that deals with the min max normalization and the reverse operation
    """
    def __init__(self, df, axis=0):
        self.name = 'minmax'
        self.axis = axis
        # get min and max according to axis
        # axis = 0 means min and max over all columns
        if self.axis == 0:
            self.min = self.get_min(df)
            self.max = self.get_max(df)
        # axis = 1 means min and max over one row
        elif self.axis == 1:
            self.min = df.min(axis = self.axis)
            self.max = df.max(axis = self.axis)
    
    def get_min(self, df):
        minimum = df.min(axis = self.axis).min()
        return minimum
        
    def get_max(self, df):
        maximum = df.max(axis = self.axis).max()
        return maximum
    
    def min_max_norm(self, df):
        # !!! potnential division by zero not handled
        if self.axis == 1:
            df_norm = df.sub(self.min, axis = 'index').div(self.max - self.min, axis = 'index')
        else:
            df_norm = (df-self.min)/(self.max - self.min)
        return df_norm
    
    def min_max_denorm(self, df):
        if self.axis == 1:
            df_denorm = df.mul(self.max - self.min, axis = 'index').add(self.min, axis = 'index')
        else:
            df_denorm = df*(self.max - self.min) + self.min
        
        return df_denorm

class norm_object_standard():
    """
    class that deals with the standard normalization and the reverse operation
    """
    def __init__(self, df):
        self.name = 'standard'
        self.mean = self.get_mean(df)
        self.standard_deviation = self.get_std(df)
    
    def get_mean(self, df):
        mean = df.mean(axis = None)
        return mean
    
    def get_std(self, df):
        standard_deviation = df.std(axis = None)
        return standard_deviation
    
    def standard_norm(self, df):
        df_norm = (df-self.mean)/(self.standard_deviation)
        return df_norm
    
    def standard_denorm(self, df):
        df_denorm = df*(self.standard_deviation) + self.mean
        return df_denorm    

class machine_learning_experimental_data(object):
    """
    this class handles the normalisation and reverse normalisation of the experimental data to be predicted on
    """
    def __init__(self, data, dic_features, feature_normalization, feature_normalization_axis='block', feature_normalization_splitting = ['dark', 'light']):
        self.data = data
        self.feature_normalization_axis = feature_normalization_axis
        self.feature_normalization_splitting = feature_normalization_splitting
        self.dic_features = dic_features
        self.features = self.select_data(dic_features)
        self.feature_normalization = feature_normalization
        self.features_normalized, self.normalization_features = self.normalize_data(self.features, self.dic_features, self.feature_normalization, axis=self.feature_normalization_axis, regex=self.feature_normalization_splitting)
        
    def select_data(self, dic):
        """
        selects data from a pandas dataframe based on the coulmn names passed in dict

        Parameters
        ----------
        dic : dictionary
            contains the names of the columns to select from data

        Returns
        -------
        data : pandas dataframe
            dataframe only containing the selected columns

        """
        data_dic = {}
        for col in dic.keys():
            data_dic[col] = self.data[col]
        data = pd.DataFrame(data_dic)
        return data
    
    def normalize_data(self, data, dic, norm, axis = 'col', regex = []):
        """
        normalize the data

        Parameters
        ----------
        data : pandas dataframe
            data to be normalised
        dic : dictionary
            dictionary containing the columns to normalise and wether or not they have to be log
        axis : string, optional
            'col' for clumn-wise, 'row' for row-wise, 'block' for block-wise. The default is 'col'.
        regex : list of strings, optional
            regular expression to be used to split data into blocks to normalise. The default is [].

        Returns
        -------
        data_normalized : pandas dataframe
            normalised data
        normalization : normalisation object
            e.g. norm_object_min_max containing info to use same normalisation on other data

        """
        data = self.log_scale(data, dic)
        if axis == 'row':
            data_normalized, normalization = self.normalize_rowwise(data, norm)
        elif axis == 'col':
            data_normalized, normalization = self.normalize_columnwise(data, norm)
        elif axis == 'block':
            data_normalized, normalization = self.normalize_blockwise(data, norm, regex)
        data_normalized = self.create_df_from_processed_data(data_normalized, data.columns)
        return data_normalized, normalization
    
    def normalize_columnwise(self, data, norm):
        """
        normalise the data column-wise axis = 0
        """
        if norm.name == 'minmax':
            df_norm = norm.min_max_norm(data)
        elif norm.name == 'standard':
            df_norm = norm.standard_norm(data)
        return df_norm, norm
    
    def normalize_rowwise(self, data, norm):
        """
        normalise the data row-wise axis = 1
        """
        if norm.name == 'minmax':
            norm = norm_object_min_max(data, axis = 1)
            df_norm = norm.min_max_norm(data)
        elif norm.name == 'standard':
            df_norm = norm.standard_norm(data)
        return df_norm, norm
    
    def normalize_blockwise(self, data, norm, regex):
        """
        normalise the data block-wise
        regex is a list with strings/sub-strings defining the blocks
        then each block gets normalised (column-wise)
        
        create dictionary with normalisation object for each regex for later use
        """
        normed_data = {}
        for exp in regex:
            data_block = data.filter(regex=exp)
            if norm[exp].name == 'minmax':
                if norm[exp].axis == 1:
                    norm[exp] = norm_object_min_max(data_block, axis = 1)
                normed_data[exp] = norm[exp].min_max_norm(data_block)
            elif norm[exp].name == 'standard':
                normed_data[exp] = norm[exp].standard_norm(data_block)
        
        data_rows = {}
        for norm_block_df in normed_data.values():
            for col in norm_block_df.columns:
                data_rows[col] = norm_block_df[col]
        
        df_norm = pd.DataFrame(data_rows)
        return df_norm, norm
    
    def log_scale(self, data, dic):
        """
        take log of data if indicated in dic
        """
        for col in data.columns:
                if dic[col]['log']=='y':
                    data[col] = np.log10(np.abs(data[col]))
        return data
    
    def create_df_from_processed_data(self, data_array, column_names):
        """
        create a pandas dataframe from array giving it column_names as column names
        """
        df = pd.DataFrame(data = data_array, columns=column_names)
        return df
