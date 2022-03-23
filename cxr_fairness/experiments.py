import numpy as np
import pandas as pd
from pathlib import Path
from cxr_fairness.data import Constants
from itertools import product
import json
from tqdm import tqdm

def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()    


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname
 
    
def add_batch_size(lst, default = 32, age = 16):
    for i in lst:
        if i['data_type'] == 'balanced' and i['protected_attr'] in ['age']:
            i['batch_size'] = age
        else:
            i['batch_size'] = default
    return lst

protected_attrs = ['sex', 'ethnicity', 'age'] 
tasks = ['No Finding', 'Pneumothorax', 'Fracture']
val_subset = 1024*12

class ERM():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'task': tasks,
            'model': ['densenet'],
            'algorithm': ['ERM'],
            'use_cache': [True],
            'val_subset': [val_subset],
            'es_metric': ['worst_roc']      
        }
        self.normal_hparams = {
            'exp_name': ['erm_baseline'],
            'data_type': ['normal'],
            'concat_group': [False]
        }
        self.normal_concat_hparams = {
            'exp_name': ['erm_baseline_concat'],
            'data_type': ['normal'],
            'concat_group': [True],
            'protected_attr': protected_attrs
        }
        self.balanced_hparams = {
            'exp_name': ['balanced'],
            'data_type': ['balanced'],
            'concat_group': [False],
            'protected_attr': protected_attrs
        }
        self.balanced_concat_hparams = {
            'exp_name': ['balanced_concat'],
            'data_type': ['balanced'],
            'concat_group': [True],
            'protected_attr': protected_attrs
        }
        self.single_group_hparams = [
            {
                'exp_name': ['single_group'],
                'data_type': ['single_group'],
                'concat_group': [False],
                'protected_attr': [attr],
                'subset_group': Constants.group_vals[attr]
            }
            for attr in Constants.group_vals
        ]
      
    def all_pairs(self, list_dict):
        return [j for i in list_dict for j in combinations({**self.base_hparams, **i})] 

    def get_hparams(self):
        return add_batch_size(self.all_pairs([self.normal_hparams, self.normal_concat_hparams, self.balanced_hparams, self.balanced_concat_hparams] + self.single_group_hparams))

class GroupDRO():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'exp_name': ['dro'],
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['GroupDRO'],
            'data_type': ['balanced'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset],  
            'task': tasks,
            'es_metric': ['worst_roc'],
            'groupdro_eta': [1.0, 0.1, 0.01]
        }
     
    def get_hparams(self):
        return  add_batch_size(combinations(self.base_hparams))

class DistMatch():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['DistMatch'],
            'data_type': ['balanced'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset],  
            'task': tasks,
            'es_metric': ['worst_roc'],
            'distmatch_penalty_weight': [100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.1]
        }

        self.mmd = {
            'exp_name': ['MMD'],
            'match_type': ['MMD']
        }

        self.meanmatch = {
            'exp_name': ['mean_match'],
            'match_type': ['mean']
        }

     
    def get_hparams(self):
        return add_batch_size(combinations({**self.base_hparams, **self.mmd}) + combinations({**self.base_hparams, **self.meanmatch}))

class SimpleAdv():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'exp_name': ['simple_adv'],
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['SimpleAdv'],
            'data_type': ['balanced'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset],  
            'task': tasks,
            'es_metric': ['worst_roc'],
            'adv_alpha': [100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.1, 0.05, 0.01]
        }
     
    def get_hparams(self):
        return add_batch_size(combinations(self.base_hparams))

class ARL():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'exp_name': ['arl'],
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['ARL'],
            'data_type': ['balanced', 'normal'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset],  
            'task': tasks,
            'es_metric': ['worst_roc']
        }
     
    def get_hparams(self):
        return add_batch_size(combinations(self.base_hparams), default = 16, age=8)


class FairALM():
    fname = 'train'
    def __init__(self):
        self.base_hparams = {
            'exp_name': ['fairalm'],
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['FairALM'],
            'data_type': ['balanced'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset], 
            'es_metric': ['worst_roc'], 
            'fairalm_eta': [1e-1, 1e-2, 1e-3],
            'task': tasks,
            'fairalm_threshold': [0.5]
        }     
     
    def get_hparams(self):
        return add_batch_size(combinations(self.base_hparams))

class JTT():
    fname = 'train'

    def __init__(self):
        erm_model_root = Path('/scratch/hdd001/home/haoran/cxr_debias/') # path to dir of trained models
        self.args_df = self.load_trained_args(erm_model_root)

        self.base_hparams = {
            'exp_name': ['jtt'],
            'val_fold': list(map(str, range(5))),
            'dataset': ['MIMIC', 'CXP'],
            'model': ['densenet'],
            'algorithm': ['JTT'],
            'data_type': ['normal'],
            'protected_attr': protected_attrs,
            'use_cache': [True],
            'val_subset': [val_subset], 
            'es_metric': ['worst_roc'], 
            'task': tasks,
            'JTT_weight': [2, 3, 5, 10, 30, 50],
            'JTT_threshold': [0.5],
            'delete_model': [True]
        }     

    def load_trained_args(self, erm_model_root):
        rows = []
        for i in tqdm(erm_model_root.glob('**/results.pkl')):   
            args_i = json.load((i.parent/'args.json').open('r'))
            args_i['JTT_ERM_model_folder'] = str(i.parent)
            if args_i['algorithm'] == 'ERM' and args_i['data_type'] == 'normal' and not args_i['concat_group']:
                rows.append(args_i)
        return pd.DataFrame(rows)
     
    def get_hparams(self):
        temp_args = add_batch_size(combinations(self.base_hparams))
        for i in temp_args:
            rows = self.args_df[(self.args_df.task == i['task']) & (self.args_df.val_fold == i['val_fold']) & (self.args_df.dataset == i['dataset'])]
            assert len(rows) == 1
            i['JTT_ERM_model_folder'] = rows.iloc[0]['JTT_ERM_model_folder']
        return temp_args

class Bootstrap():
    fname = 'bootstrap'

    def get_hparams(self):
        configs_path = '/scratch/hdd001/home/haoran/cxr_debias/selected_configs.pkl' # path to config output by notebooks/get_best_model_configs.ipynb
        selected_configs = pd.read_pickle(configs_path)
        out_dir = Path('/ssd003/home/haoran/cxr_debias_boot/') # path to output bootstrap results

        all_hparams = []
        for dataset in np.flip(selected_configs.dataset.unique()):
            for model in selected_configs.model.unique():
                for task in selected_configs.task.unique():          
                    select_metric = 'worst_roc'
                    all_hparams.extend(combinations({
                        'exp_name': ['bootstrap'],
                        'configs_path': [configs_path],
                        "out_path": [str(out_dir/f'{dataset}_{model}_{task.lower().replace(" ", "_")}_{select_metric}_boot.pkl')],
                        "dataset": [dataset],
                        'model': [model],
                        'task': [task],
                        'select_metric': [select_metric],
                        "n_boot": [250],
                        'baseline_exp_name': ['balanced']
                    }))

                    select_metric = 'vary_lambda_exp'
                    if dataset == 'MIMIC':
                        for exp_starts_with in ['MMD', 'mean_match', 'simple_adv']:
                            all_hparams.extend(combinations({
                                'exp_name': ['bootstrap'],
                                'configs_path': [configs_path],
                                "out_path": [str(out_dir/f'{dataset}_{model}_{task.lower().replace(" ", "_")}_{select_metric}_{exp_starts_with}_boot.pkl')],
                                "dataset": [dataset],
                                'model': [model],
                                'task': [task],
                                'select_metric': [select_metric],
                                "n_boot": [250],
                                'exp_starts_with': [exp_starts_with]
                            }))

        return all_hparams

