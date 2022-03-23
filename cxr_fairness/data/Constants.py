import os
from collections import defaultdict

#-------------------------------------------
image_paths = {
    'MIMIC': '/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR-JPG', # MIMIC-CXR
    'CXP': '/scratch/hdd001/projects/ml4h/projects/CheXpert/', # CheXpert
}

CXP_details = "/scratch/hdd001/projects/ml4h/projects/CheXpert/CHEXPERT DEMO.xlsx"

cache_dir = '/scratch/ssd001/home/haoran/projects/IRM_Clinical/cache'
#-------------------------------------------

df_paths = {
    dataset: os.path.join(image_paths[dataset], 'cxr_fairness', f'preprocessed.csv')
    for dataset in image_paths 
}

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema']
take_labels_all = take_labels + ['Enlarged Cardiomediastinum', 'Airspace Opacity', 'Lung Lesion', 'Pleural Other', 'Fracture', 'Support Devices']

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

ethnicity_mapping = defaultdict(lambda:2)
ethnicity_mapping['WHITE'] = 0
ethnicity_mapping['BLACK/AFRICAN AMERICAN'] = 1

group_vals = {
    'sex': ['M', 'F'],
    'ethnicity': [0, 1, 2],
    'age': ["18-40", "40-60", "60-80", "80-"]
}