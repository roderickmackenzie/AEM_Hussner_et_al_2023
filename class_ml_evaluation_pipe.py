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
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import logging.config
import pandas as pd

logger_path = os.path.join(os.path.expanduser("~"), 'ML_code', 'logger')
logging.config.fileConfig(fname=os.path.join(logger_path, 'file.conf'), disable_existing_loggers=True)
# Get the logger specified in the file
logger = logging.getLogger('fileLogger')
logger.setLevel('INFO')

class ml_plot(object):
    
    def __init__(self, dataset_name, model_name, save_plots=False, sub_dir=''):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_plots = save_plots
        self.sub_dir = sub_dir
        
    def plot_loss(self, history, save_dir = ''):
        """
        plots the loss over training epch
        
        Parameters
        ----------
        history : tensorflow model.fit() object 
            object returned by model.fit method. containing history of the training process including validation scores and losses
        save_dir : string, optional
            name of a directory other than the default os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir

        Returns
        -------
        fig : matplotlib figure
            figure showing loss vs. training epoch
        ax : matplotlib axes

        """
        fig, ax= plt.subplots()
        ax.plot(history.history['loss'], label='loss')
        ax.plot(history.history['val_loss'], label='val_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        if save_dir == '':
            path = os.path.join(os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir)
        else:
            path = save_dir
            
        if self.save_plots == True:
            try:
                os.makedirs(path)
            except:
                logger.info('Overwriting files')
            fig.savefig(os.path.join(path, "loss.svg"))
        return fig,ax
    
    def get_error_vectors(self, test_labels, test_predictions, dic, denorm=False):
        """
        if network is trained on normalized data test_predictions have to be denormalized by passing denorm
        
        Parameters
        ----------
        test_labels : pandas df
            all values for all labels/truth of test set
        test_predictions : list of dim (number of labels, number of samples in test set)
            all values for all predictions on the test set
        dic : dictionary of dictionary
            contains all labal names and as dict. human readable name, unit, if log-data or not
        denorm : normalization object from class_ml_data_pipe.py, optional
            the denormalization function already adapted to the train_label data. The default is False.

        Returns
        -------
        error_vec : dictionary
            keys: parameter in test_label, values: (np.array with truth, np.array with prediction) arrays dim (number of samples in test set)

        """
        error_vec = {}
        #parameters = [col for col in test_labels.columns]
        parameters = [col[0] for col in dic.items()]
        
        if denorm == False:
            for i, parameter in enumerate(parameters):
                test_label = test_labels[parameter].to_numpy()
                test_prediction = tf.gather(test_predictions, i, axis=1).numpy()
                error_vec[parameter] = [test_label, test_prediction]
        else:
            # invert normalization
            if denorm.name == 'minmax':
                test_pred_denorm = denorm.min_max_denorm(test_predictions)
                test_labels = denorm.min_max_denorm(test_labels)
            elif denorm.name == 'standard':
                test_pred_denorm = denorm.standard_denorm(test_predictions)
                test_labels = denorm.standard_denorm(test_labels)
            else:
                logger.error('Error. Undefined normalizaton type!')
                
            for i, parameter in enumerate(parameters):
                # convert results of inverse normalization into a numpy array
                test_label = tf.gather(test_labels, i, axis=1).numpy()
                test_prediction = tf.gather(test_pred_denorm, i, axis=1).numpy()
                # invert log10()
                if dic[parameter]['log'] == 'y':
                    test_prediction = pow(10,test_prediction)
                    test_label = pow(10,test_label)
                error_vec[parameter] = [test_label, test_prediction]
        return error_vec
    
    def plot_predictions(self, error_vec, dic, save_dir=''):
        """
        plot confusion matrices of the errors of the test prediction
        
        Parameters
        ----------
        error_vec : dictionary
            keys: parameter in test_label, values: (np.array with truth, np.array with prediction), arrays dim (number of samples in test set)
        dic : dictionary
            dic with parameters as keys and alternative name, unit and log y/n as values
       save_dir : string, optional
           name of a directory other than the default os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir

        Returns
        -------
        plots : dictionary
            keys: name of parameters, values: (fig, ax) provided by plot_pred
        """
        plots = {}
        for item in error_vec.items():
            parameter = item[0]
            test_label = item[1][0]
            test_prediction = item[1][1]
            plots[parameter] = self.plot_pred(test_label, test_prediction, dic[parameter])
            
        if save_dir == '':
            path = os.path.join(os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir)
        else:
            path = save_dir
            
        if self.save_plots == True:
            try:
                os.makedirs(path)
            except:
                logger.info('Overwriting files')
            df = pd.DataFrame(error_vec, index = ['label', 'prediction'])
            df.to_csv(os.path.join(path,'error_vec.csv'))
            for plot in plots:
                plots[plot][0].savefig(os.path.join(path, dic[plot]['title']+".svg"))
        return plots
    
    def plot_pred(self, test_label, test_prediction, plot_parameters):
        """
        plot one single confusion matrix for one target of the test prediction
        
        Parameters
        ----------
        test_label : numpy array of dim (number of samples of test set)
            truth/labels of the test set
        test_prediction : numpy array of dim (number of samples of test set)
            prediction on the test set using the test features
        plot_parameters : list
            list with (parameter name, unit, log y/n) 

        Returns
        -------
        fig : matplotlib figure
            confusion matrix prediction vs truth of test set
        ax : matplotlib axes

        """
        fig,ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(test_label, test_prediction, s=0.5)
        try:
            #ax.set_title(plot_parameters['plot_name'] + '\n[' + plot_parameters['unit'] + ']')
            ax.set_title(plot_parameters['title'] + '\n[' + plot_parameters['unit'] + ']')
        except:
            ax.set_title(plot_parameters)

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        try:
            if plot_parameters['log'] == 'y':
                ax.set_xscale('log')
                ax.set_yscale('log')
        except:
            pass
        return fig, ax
    
    def plot_errorhist(self, error, plot_parameters):
        """
        plot histogram of the error truth-prediction for one target
        
        Parameters
        ----------
        error : np.array dim (number of samples in test set)
            error truth - prediction
        plot_parameter : list
            list with (parameter name, unit, log y/n) 

        Returns
        -------
        fig : matplotlib figure
            figure of histogram of the error truth-prediction
        ax : matplotlib axes
        """
        fig, ax = plt.subplots()
        ax.hist(error, bins=25)
        try:
            ax.set_xlabel('Prediction Error [' + plot_parameters['unit'] + ']')
        except:
            ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        try:
            ax.set_title('Error of prediction for ' + plot_parameters['title'])
        except:
            ax.set_title('Error of prediction for ' + plot_parameters)
        return fig,ax
    
    def plot_error_histograms(self, error_vec, dic, save_dir=''):
        """
        plot histogram of the error truth-prediction for all targets in error_vec
        
        Parameters
        ----------
        error_vec : dictionary
            keys: parameter in test_label, values: (np.array with truth, np.array with prediction) arrays dim (number of samples in test set)
        dataset_name : string
            name of the dataset that was used
        model_name : string
            name of the experiment/sim
        dic : dictionary
            dic with parameters as keys and alternative name, unit and log y/n as values
        save_plots : boolean, optional
            if the plots should be saved or not The default is True.
        save_dir : string, optional
            name of a directory other than the default os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir

        Returns
        -------
        plots : dictionary
            keys: name of parameters, values: (fig, ax) provided by plot_error_hist
        """
        plots = {}
        for item in error_vec.items():
            parameter = item[0]
            test_label = item[1][0]
            test_prediction = item[1][1]
            error = test_label - test_prediction
            print('label', test_label)
            print('prediction', test_prediction)
            print('----error here -----', error)
            plots[parameter] = self.plot_errorhist(error, dic[parameter])
        
        logger.debug(plots)
        
        if save_dir == '':
            path = os.path.join(os.path.expanduser('~'), 'ML', 'plots', 'NN', self.dataset_name, self.model_name, self.sub_dir)
        else:
            path = save_dir
            
        if self.save_plots == True:
            try:
                os.makedirs(path)
            except:
                logger.info('Overwriting files')
            for plot in plots:
                plots[plot][0].savefig(os.path.join(path ,dic[plot]['title']+"_errorhist.svg"))
        return plots
