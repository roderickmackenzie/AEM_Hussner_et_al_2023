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
sys.path.append(os.path.join(os.path.expanduser('~'), 'AEM_Hussner_et_al_2023'))
import training as training
import predict as predict

def main():
    dataset_name = 'training_data_pm6dty6'
    # to identify a feature = ML token in oghma
    feature_regex = 'vec'
    # path of the data set
    dataset_path = os.path.join(os.path.expanduser("~"), 'AEM_Hussner_et_al_2023', 'data')
    # path to where the experimental data as csv file is
    device_path = os.path.join(os.path.expanduser('~'), 'AEM_Hussner_et_al_2023')
    devices = ['example_input']
    # light intensities at which the JV curves are available\ simulated
    intensities = [ 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    # voltages at which jv curves in oghma were simulated
    
    for device in devices:
        model_path = os.path.join(os.path.expanduser('~'), 'AEM_Hussner_et_al_2023', device)
        model_name = 'mu_geom_jsc'
        save_p = 'True'
        save_m = 'True'
        #training.main([dataset_name, dataset_path, os.path.join(model_path, dataset_name), model_name, save_m, save_p, feature_regex, intensities, device_path, device])
        predict.main([dataset_name, dataset_path, model_name, model_path, save_p, feature_regex, intensities, device_path, device])

if __name__ == "__main__":
    main()
