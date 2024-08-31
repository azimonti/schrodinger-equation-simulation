#!/usr/bin/env python3
'''
/********************************/
/* pickle_joblib_convert_s2d.py */
/*          Version 1.0         */
/*           2024/08/31         */
/********************************/
'''
import os
import joblib
import pickle
# it should be run from the root directory for this import
from schrodinger_2d import WavepacketSimulation

DIR = "DIRPATH"
# Set to True if converting from pickle to joblib
from_pickle = False


def convert_files(directory, from_pickle):
    ext = '.pkl' if from_pickle else '.joblib'
    new_ext = '.joblib' if from_pickle else '.pkl'

    files = ['config_s2d', 'data_s2d']

    for file_name in files:
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


convert_files(DIR, from_pickle)
