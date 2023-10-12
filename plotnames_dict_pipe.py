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
import pandas as pd
import json
import re

def create_names_manually(df):
    """
    creates dictionary with alternative names for columns in df together with unit and log y/n
    only considers columns without 'vec' in name (so only targets)
    Parameters
    ----------
    df : pandas dataframe
        df with the features and targets

    Returns
    -------
    plot_labels : dict
        dict with alternative names, unit and log y/n

    """
    df_features = df.filter(regex='vec')
    
    df_labels = df.drop(columns = df_features.columns)
    
    plot_labels = {}
    for column in df_labels.columns:
        inp_name = input('new name for '+ column)
        inp_unit = input('unit for '+ column)
        print('log? y/[n]')
        inp_log = input("log? y/[n]:") or "n"
        plot_labels[column] = {'plot_name': inp_name, 'unit': inp_unit, 'log': inp_log, 'title': inp_name}
    return plot_labels

def save_dict_as_json(dic, path, name):
    """
    saves dictionary as .json with name in path
    Parameters
    ----------
    dic : dictionary
        dictionary (here e.g. created by create_names_manually)
    path : string
        os path where the file is saved
    name : string
        name of file to be saved

    Returns
    -------
    None
    """
    with open(os.path.join(path, name+'.json'), 'w+') as fp:
        json.dump(dic, fp)
    return None

def load_json(path, name):
    """
    Parameters
    ----------
    path : string
        path where the file is loaded from
    name : string
        name of json-file (without file ending)

    Returns
    -------
    data : json-object
        contains data of the json file

    """
    with open(os.path.join(path, name+'.json'), 'r') as fp:
        data = json.load(fp)
    return data

def replace_df_col_names(df, dic):
    """
    Parameters
    ----------
    df : pandas dataframe
        original dataframe with columns to be replaced
    dic : dictionary
        dictionary with original and alternative names

    Returns
    -------
    df : pandas dataframe
        dataframe where the column names are replaced by their alternative names taken from dic
        if there is no  alternative name, do not replace

    """
    for column in df.columns:
        try:
            if dic[column]['plot_name'] != '':
               df = df.rename(columns={column: dic[column]['plot_name']})
        except:
            pass
    return df

def create_plotfile_from_template(template, new_parts):
    datanew = {}
        
    p = re.compile('light')
    for part in new_parts:
        for key in template:
            new_key = p.sub('light_' + f'{part}', key)
            datanew[new_key] = template[key].copy()
            #print(datanew[new_key], datanew[new_key]['plot_name'] + ' @ ' + part + ' suns')
            datanew[new_key]['title'] = datanew[new_key]['plot_name'] + ' @ ' + f'{part}' + ' suns'
            print(new_key)
    return datanew

def add_title_to_json(template):
    for key in template:
        template[key]['title'] = template[key]['plot_name']
    return template

def main():
    # dataset_name = 'sn21full'#sys.argv[1] #"dataset_vectors"
    # # read csv file intopandas dataframe
    # df = pd.read_csv("datasets/"+dataset_name+".csv")
    # df.drop(columns=["Unnamed: 0", "samples"], inplace=True)
    # print(df)
    
    # plot_labels = create_names_manually(df)
    # save_dict_as_json(plot_labels, 'datasets/', dataset_name+'plots')
    
    # plot_labels = load_json('datasets/', dataset_name+'plots')
    # print(plot_labels)
    
    # print(plot_labels['light_voc']['unit'])
    # df = replace_df_col_names(df, plot_labels)
    # print(df)
    
    path = './datasets/'
    name = 'template'
    data = load_json(path, name)
    
    datanew = {}
    new_parts =[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    datanew = create_plotfile_from_template(data, new_parts)
    #datanew = add_title_to_json(data)
    print(datanew)
    save_dict_as_json(datanew, path, 'sn21_intensity_2plots')
    
if __name__ == "__main__":
    main()
