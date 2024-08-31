#!/usr/bin/env python3
'''
/*****************************/
/* pickle_joblib_convert.py  */
/*        Version 1.0        */
/*         2024/08/31        */
/*****************************/
'''
import os
import joblib
import pickle
# it should be run from the root directory for these imports
from double_slit_2d import DoubleSlitSimulation
from schrodinger_2d import WavepacketSimulation

DIR = "DIRPATH"
# Set to True if converting from pickle to joblib
from_pickle = False

def convert_files(directory, from_pickle, file_names):
    ext = '.pkl' if from_pickle else '.joblib'
    new_ext = '.joblib' if from_pickle else '.pkl'

    for file_name in file_names:
        file_path = os.path.join(directory, f'{file_name}{ext}')

        if os.path.exists(file_path):
            if from_pickle:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                joblib.dump(data, file_path.replace(ext, new_ext))
            else:
                data = joblib.load(file_path)
                with open(file_path.replace(ext, new_ext), 'wb') as f:
                    pickle.dump(data, f)

# File names for both double_slit_2d and schrodinger_2d
files = ['config_ds2d', 'data_ds2d', 'config_s2d', 'data_s2d']

# Convert all files in a single loop
convert_files(DIR, from_pickle, files)
