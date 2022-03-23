from cxr_fairness.data import Constants
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from pathlib import Path
import random

def process_MIMIC(split, only_frontal, return_all_labels = True):  
    copy_subjectid = split['subject_id']     
    split = split.drop(columns = ['subject_id']).replace(
            [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
             'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
             '90-100'],
            [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
             'DIVORCED/SEPARATED', '18-40', '18-40', '18-40', '18-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    
    split['subject_id'] = copy_subjectid.astype(str)
    split['study_id'] = split['study_id'].astype(str)
    split['age'] = split["age_decile"]
    split['sex'] = split["gender"]
    split = split.rename(
        columns = {
            'Pleural Effusion':'Effusion',  
            'Lung Opacity': 'Airspace Opacity' 
        })
    split['path'] = split['path'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['MIMIC'], x))
    if only_frontal:
        split = split[split.frontal]
    
    split['ethnicity'] = split['ethnicity'].map(Constants.ethnicity_mapping)
    split['env'] = 'MIMIC'  
    split = split[split.age != 0]
    
    return split[['subject_id','path','sex',"age", 'ethnicity', 'env', 'frontal', 'study_id', 'fold_id'] + Constants.take_labels +
            (['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices'] if return_all_labels else [])]

def process_CXP(split, only_frontal, return_all_labels = True):
    def bin_age(x):
        if 0 <= x < 40: return '18-40'
        elif 40 <= x < 60: return '40-60'
        elif 60 <= x < 80: return '60-80'
        else: return '80-'

    split['Age'] = split['Age'].apply(bin_age)
    
    copy_subjectid = split['subject_id'] 
    split = split.drop(columns = ['subject_id']).replace([[None], -1, "[False]", "[True]", "[ True]"], 
                            [0, 0, 0, 1, 1])
    
    split['subject_id'] = copy_subjectid.astype(str)
    split['Sex'] = np.where(split['Sex']=='Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex']=='Male', 'M', split['Sex'])    
    split = split.rename(
        columns = {
            'Pleural Effusion':'Effusion',
            'Lung Opacity': 'Airspace Opacity',
            'Sex': 'sex',
            'Age': 'age'  
        })
    split['path'] = split['Path'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['CXP'], x))
    split['frontal'] = (split['Frontal/Lateral'] == 'Frontal')
    if only_frontal:
        split = split[split['frontal']]
    split['env'] = 'CXP'
    split['study_id'] = split['path'].apply(lambda x: x[x.index('patient'):x.rindex('/')])    

    return split[['subject_id','path','sex',"age", 'env', 'frontal','study_id', 'fold_id', 'ethnicity'] + Constants.take_labels +
                (['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices'] if return_all_labels else [])]


def get_process_func(env):
    if env == 'MIMIC':
        return process_MIMIC
    elif env == 'CXP':
        return process_CXP
    else:
        raise NotImplementedError        


