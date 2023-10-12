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
import keras as K
from keras.layers import Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gc

class build_keras_model(object):
    """
    Base class for neural network creation
    https://www.kaggle.com/code/varun2145/how-to-iterate-over-keras-layer-as-a-parameter
    creation, training and saving of the model
    """
    
    def __init__(self, layers, activation, opt, init, input_dim, output_dim, patience, 
                 is_batchnorm, loss_func, metrics, print_model_summary, verbose, kernel_regularizer=None, dropout=False):
        
        # type of Keras model
        self.model = K.models.Sequential()
        # list of layers with numbers of nodes, excluding input and output layers e.g. [200, 50, 10]
        self.layers = layers
        # activation function for the neurons e.g. Relu or sigmoid
        self.activation = activation
        # the Optimizer for the training process e.g. ADAM optimizer
        self.opt = opt
        # initialisation of the weights
        self.init = init
        # dimension of the input vector
        self.input_dim = input_dim
        # dimension of the output vector
        self.output_dim = output_dim
        # patience, how long training will be continued once the target error is reached
        self.patience = patience
        # batchnormalization 0=No, 1=Yes
        self.is_batchnorm = is_batchnorm
        # the loss function to use e.g. MSE
        self.loss_func = loss_func
        # print model summary 0=No, 1=Yes
        self.prmodsum = print_model_summary
        # metric to use for model optimization, e.g. validation error or R2 score
        self.metrics = metrics
        # print information on training 0=No, 1=Yes
        self.verbose = verbose
        # regularization, e.g. L2 norm regularization
        self.regularizer = kernel_regularizer
        # list of fraction of nodes to not train each epoch, e.g. [0.5, 0.3, 0.1]
        self.dropout = dropout

    def create_model(self):
        """
        creates the model from the initialised parameters

        Returns
        -------
        None.

        """
        
        # for each node in layers add this as a dense layer and apply dropout and batchnorm
        for i, nodes in enumerate(self.layers):
            if i==0:
                self.model.add(K.layers.Dense(nodes,input_dim=self.input_dim,kernel_initializer=self.init, kernel_regularizer=self.regularizer))
                self.model.add(Activation(self.activation))
                if self.is_batchnorm == 1:
                    self.model.add(BatchNormalization())
                if len(self.dropout) != 0:
                    self.model.add(K.layers.Dropout(self.dropout[i]))
            else:
                self.model.add(K.layers.Dense(nodes,kernel_initializer=self.init, kernel_regularizer=self.regularizer))
                self.model.add(Activation(self.activation))
                if self.is_batchnorm == 1:
                    self.model.add(BatchNormalization())
                if len(self.dropout) != 0:
                    self.model.add(K.layers.Dropout(self.dropout[i]))

        # add output layer
        self.model.add(K.layers.Dense(self.output_dim))
        self.model.add(Activation(self.activation)) # Note: no activation beyond this point
        
        if self.prmodsum == 1:
            print(self.model.summary())

        # create the model
        self.model.compile(optimizer=self.opt, loss=self.loss_func, metrics=self.metrics)
        return
    
    def fit_model(self, X, y, validation_split, batch_size, epochs, callbacks_list):
        """
        Train the model

        Parameters
        ----------
        X : pandas dataframe
            features
        y : pandas dataframe
            targets
        validation_split : float
            fraction of the data set that will be used for the validation set
        batch_size : int
            batch size for training
        epochs : int
            number of training epachs
        callbacks_list : list
            list with callback to the model e.g. for early stopping

        Returns
        -------
        TYPE keras model 
            fitted/trained model

        """
        pt = self.patience
        vb = self.verbose
        
        return self.model.fit(X, y, validation_split = validation_split, 
                                                batch_size=batch_size,
                                                epochs=epochs, 
                                                verbose=vb, 
                                                callbacks=callbacks_list,
                                                shuffle = True)
    
    def save_model(self, name, path):
        """
        save the model
        
        Parameters
        ----------
        name : string
            name of the model
        path : string
            string of path/directory to save the model at

        Returns
        -------
        None.

        """
        try:
            os.makedirs(path)
        except:
            print('Overwrite model')
            
        self.model.save(os.path.join(path, name))
    
    def __del__(self): 
        del self.model
        gc.collect()
        if self.prmodsum == 1:
            print("Keras Destructor called") 
