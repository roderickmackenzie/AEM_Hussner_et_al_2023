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
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

sys.path.append(os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023_model_data'))
import class_nnmodel_pipe as nn
import class_ml_data_pipe as mldata
import class_ml_evaluation_pipe as ev

from plotnames_dict_pipe import load_json

import logging
import logging.config
logger_path = os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023_model_data', 'logger')
logging.config.fileConfig(fname=os.path.join(logger_path, 'file.conf'), disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger('fileLogger')

def choose_targets(targets, dic):
    dic_targets = {}
    for t in targets:
        dic_targets[t] = dic[t]
    return dic_targets

def create_features_dic(features):
    dic_features = {}
    regexp_log = re.compile('dark')
    regexp_lin = re.compile('light_')
    for f in features:
        if regexp_log.search(f):
            dic_features[f] = {'log': 'y'}
        elif regexp_lin.search(f):
            dic_features[f] = {'log': 'n'}
    return dic_features

def convert_string_to_bool(s): 
    if s.lower() == "true": 
        return True
    return False

def main(args):
    dataset_name = args[0]
    # save the model?
    dataset_path = args[1]
    save_m = convert_string_to_bool(args[4])
    model_save_path = args[2]
    model_name = args[3]
    #save the plots?
    save_p = convert_string_to_bool(args[5])
    # which string denotes a feature vector
    feature_regex = args[6]
    
    device_path = args[8]
    device = args[9]
    plot_save_path = os.path.join(device_path, device, model_name)
    # read csv file intopandas dataframe
    df = pd.read_csv(os.path.join(dataset_path, dataset_name+".csv"))
    df = df[(df != -1).all(axis=1)]
    print('----------------------------df------------------------', df)
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        df.drop(columns=["samples"], inplace=True)
    except:
        pass
    
    intensities = args[7]
    targets = []
    for i in intensities:
        targets.append(f'light_{i}_{model_name}' )
    logger.debug(targets)
        
    features = df.filter(regex=f'{feature_regex}')
    print(features)
    splitting = []
    for i in intensities:
        splitting.append(f'light_{i}_{feature_regex}')
    splitting.append(f'dark_{feature_regex}')
    
    try:
        plot_labels = load_json(os.path.join(dataset_path), dataset_name+'plots')
        logger.info('opened alt names file')
    except:
        logger.info('no alternative column names found')
    
    dic_features = create_features_dic(features)
    print('----------------', dic_features)
    dic_targets = choose_targets(targets, plot_labels)
    dataset = mldata.machinelearning_data_struct(df, dic_features, dic_targets, target_normalization_axis='col', feature_normalization_axis='block', feature_normalization_splitting=splitting)
    dataset.dataset_name = dataset_name
    
    test_results = {}
    test_results['nn_model'] = [0,-1]
    while test_results['nn_model'][1]<=0:
        # set up the model with keras.Sequential using the normalizer and respective layers
        layer_nodes = [200, 50,50,50]
        activation = 'sigmoid'
        opt = tf.keras.optimizers.Adam(0.01)
        init = 'glorot_uniform'
        input_dim = len(dataset.features.columns)
        output_dim = len(targets)
        patience = 26
        is_batchnorm = 0
        loss_func = 'mse'
        metrics = tfa.metrics.r_square.RSquare()
        print_model_summary = 0
        verbose = 1
        dropout = []
        regularization = None#regularizers.l2(0.0001)
        nn_model = nn.build_keras_model(layer_nodes, activation, opt, init, input_dim, output_dim, patience, is_batchnorm, loss_func, metrics, print_model_summary, verbose, dropout=dropout, kernel_regularizer=regularization)
        nn_model.create_model()
        
        # create callbacks e.g. for early stopping
        callbacks = []
        callbacks = [keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-12 less"
            min_delta=1e-12,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=10,
            verbose=1,
            ),]
        
        if save_m == True:
            checkpoint_filepath = '/tmp/checkpoint/'
            try:
                os.makedirs(checkpoint_filepath)
            except:
                logger.info('Overwrite model') 
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
            callbacks.append(model_checkpoint_callback)
            logger.info('------------------------save-----------------------------')
        
        start = datetime.now()
        
        logger.debug(('features', dataset.features_training))
        logger.debug(('targets', dataset.targets_training))
        
        history = nn_model.fit_model(dataset.features_training, dataset.targets_training, 0.2, 32, 500, callbacks)
        # plot the loss(epoch)
        evaluation = ev.ml_plot(dataset.dataset_name, model_name, save_plots=save_p)
        evaluation.plot_loss(history, save_dir = plot_save_path)
        
        # store performance of model at the end of training
        test_results = {}
        test_results['nn_model'] = nn_model.model.evaluate(
        dataset.features_test, dataset.targets_test, verbose=0)
        logger.debug(nn_model.model.metrics_names)
        logger.debug(('after training', test_results['nn_model']))
        
        if save_m == True:
            nn_model.model.load_weights(checkpoint_filepath)
        
        #store performance of best model
        test_results['nn_model'] = nn_model.model.evaluate(
        dataset.features_test, dataset.targets_test, verbose=0)
        
        # make predictions and plot them after denormalization
        test_predictions = pd.DataFrame(nn_model.model.predict(dataset.features_test))
        total_t = datetime.now() - start
        
        error_vector = evaluation.get_error_vectors(dataset.targets_test, test_predictions, dataset.dic_targets, denorm=dataset.normalization_targets)
        
        plots = evaluation.plot_predictions(error_vector, dataset.dic_targets, save_dir = plot_save_path)
        
        # plot error histogram
        error_plots = evaluation.plot_error_histograms(error_vector, dataset.dic_targets, save_dir = plot_save_path)
        
        logger.debug(('best model', test_results['nn_model']))
        logger.debug(total_t)
        
        # save model
        if save_m == True:
            nn_model.save_model(model_name, model_save_path)
        
        plt.autoscale()
    
    logger.info('-------------------------- end of training --------------------------')
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

