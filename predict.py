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
import pickle

import tensorflow as tf
import tensorflow_addons as tfa

sys.path.append(os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023_model_data', 'modules'))
from opkm_plot import make_fig
sys.path.append(os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023_model_data'))
import class_ml_data_pipe as mldata
from plotnames_dict_pipe import load_json

import logging
import logging.config
logger_path = os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023_model_data', 'logger')
logging.config.fileConfig(fname=os.path.join(logger_path, 'file.conf'), disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger('fileLogger')
logger.setLevel('INFO')

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
    dataset_path = args[1]
    
    logger.info(dataset_name)
    
    model_name = args[2]
    model_path = args[3]
    # save the model?
    model_save_path = os.path.join(model_path, dataset_name, model_name)
    #save the plots?
    save_p = convert_string_to_bool(args[4])
    # which string denotes a feature vector
    feature_regex = args[5]
    # read csv file intopandas dataframe
    df = pd.read_csv(os.path.join(dataset_path, dataset_name+".csv"))
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        df.drop(columns=["samples"], inplace=True)
    except:
        pass
    
    intensities = args[6]
    targets = []
    for i in intensities:
        targets.append(f'light_{i}_{model_name}' )
    logger.debug(targets)
    
    features = df.filter(regex=f'{feature_regex}')
    splitting = []
    for i in intensities:
        splitting.append(f'light_{i}_{feature_regex}')
    splitting.append(f'dark_{feature_regex}')
    
    try:
        plot_labels = load_json(os.path.join(dataset_path), dataset_name+'plots')
        logger.info('opened alt names file')
    except:
        logger.info('no alternative column names found')
    
    # this is the experimental data
    dic_features = create_features_dic(features)
    dic_targets = choose_targets(targets, plot_labels)
    dataset = mldata.machinelearning_data_struct(df, dic_features, dic_targets, target_normalization_axis='col', feature_normalization_axis='block', feature_normalization_splitting=splitting)
    dataset.dataset_name = dataset_name
    
    device_path = args[7]
    device = args[8]
    path = os.path.join(os.path.expanduser('~'), device_path)
    dataset_name = dataset_name.strip('_noise')
    file = f'{dataset_name}_{device}_exp.csv'
    df2 = pd.read_csv(os.path.join(path, file))
    logger.debug(df2)
    dic_features = create_features_dic(features)
    dic_targets = choose_targets(targets, plot_labels)
    dataset_exp = mldata.machine_learning_experimental_data(df2, dic_features, dataset.normalization_features, feature_normalization_axis='block', feature_normalization_splitting=splitting)
    dataset_exp.dataset_name = dataset_name
    
    model = tf.keras.models.load_model(model_save_path, custom_objects={"r_square": tfa.metrics.r_square.RSquare()})
    model.summary()
    
    predictions =  pd.DataFrame(model.predict(dataset_exp.features_normalized))
    if plot_labels[targets[0]]['log'] == 'y':
        denormed_predictions = pow(10, dataset.normalization_targets.min_max_denorm(predictions))
    else:
        denormed_predictions = dataset.normalization_targets.min_max_denorm(predictions)
        
    df_pred = pd.DataFrame(denormed_predictions)
    logger.debug(('prediction ------------------', df_pred))
    
    if save_p == True:
        try:
            os.makedirs(path)
        except:
            logger.info('Overwriting files')
        df_pred.to_csv(os.path.join(model_save_path, f'prediction_experimental_{dataset_name}_{device}.csv'))
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
