import cxr_fairness.data.Constants as Constants
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import json
import random
from cxr_fairness.data.preprocess.validate import validate_all
from sklearn.model_selection import KFold

def split(df, n_splits_train = 5, random_seed = 42):
    skf = KFold(n_splits = n_splits_train + 1, random_state = random_seed, shuffle = True)
    df['fold_id'] = ''
    for c, (_, fold_index) in enumerate(skf.split(df.index)):
        if c == n_splits_train:
            df.loc[fold_index, 'fold_id'] = 'test'
        else:
            df.loc[fold_index, 'fold_id'] = str(c)
    return df

def preprocess_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    out_folder = img_dir/'cxr_fairness'
    out_folder.mkdir(parents = True, exist_ok = True)  

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    ethnicities = pd.read_csv(img_dir/'admissions.csv.gz').drop_duplicates(subset = ['subject_id']).set_index('subject_id')['ethnicity'].to_dict()
    patients['ethnicity'] = patients['subject_id'].map(ethnicities)
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 101, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)
    
    df = split(df.reset_index(drop = True))
    df.to_csv(out_folder/"preprocessed.csv", index=False)

def preprocess_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    out_folder = img_dir/'cxr_fairness'
        
    if (img_dir/'CheXpert-v1.0'/'train.csv').is_file():
        df = pd.concat([pd.read_csv(img_dir/'CheXpert-v1.0'/'train.csv'), 
                        pd.read_csv(img_dir/'CheXpert-v1.0'/'valid.csv')],
                        ignore_index = True)
    elif (img_dir/'CheXpert-v1.0-small'/'train.csv').is_file(): 
        df = pd.concat([pd.read_csv(img_dir/'CheXpert-v1.0-small'/'train.csv'),
                        pd.read_csv(img_dir/'CheXpert-v1.0-small'/'valid.csv')],
                        ignore_index = True)
    elif (img_dir/'train.csv').is_file():
        raise ValueError('Please set Constants.image_paths["CXP"] to be the PARENT of the current'+
                ' directory and rerun this script.')
    else:
        raise ValueError("CheXpert files not found!")

    out_folder.mkdir(parents = True, exist_ok = True)  
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    df = split(df.reset_index(drop = True))
    details = pd.read_excel(Constants.CXP_details, engine = 'openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)    
    
    df = pd.merge(df, details, on = 'subject_id', how = 'inner')

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    df.reset_index(drop = True).to_csv(out_folder/"preprocessed.csv", index=False)

if __name__ == '__main__':
    print("Validating paths...")
    validate_all()
    print("Preprocessing MIMIC-CXR...")
    preprocess_mimic()
    print("Preprocessing CheXpert...")
    preprocess_cxp()
