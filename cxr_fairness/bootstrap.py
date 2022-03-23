import numpy as np
import pandas as pd
from cxr_fairness.metrics import StandardEvaluator, FairOVAEvaluator
import argparse
from tqdm import tqdm, trange
from cxr_fairness.data import data
from cxr_fairness.data import Constants
from cxr_fairness.postprocess import CalibratedEqualizedOddsPostProcessor, EqualizedOddsPostProcessor
from pathlib import Path
import random
import json
import pickle
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required = True)
parser.add_argument("--configs_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument('--dataset', type = str)
parser.add_argument('--model', type = str)
parser.add_argument('--task', type = str)
parser.add_argument('--select_metric', type = str)
parser.add_argument("--n_boot", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--thresholds", type=float, nargs='+', default = [0.75, 0.5, 0.25, 0.1])
parser.add_argument("--postprocess_thresholds", type=float, nargs='+', default = [])
parser.add_argument("--sens_at_spec", type=float, nargs='+', default = np.arange(0.1, 1., 0.1).tolist())
parser.add_argument("--baseline_exp_name", type=str)
parser.add_argument("--exp_starts_with", type=str)
parser.add_argument('--n_jobs', default = -1, type = int)
parser.add_argument('--debug', action = 'store_true')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
    
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

configs = pd.read_pickle(args.configs_path)
if args.dataset is not None:
    configs = configs[configs.dataset == args.dataset]

if args.model is not None:
    configs = configs[configs.model == args.model]

if args.task is not None:
    configs = configs[configs.task == args.task]

if args.select_metric is not None:
    configs = configs[configs.select_metric == args.select_metric]

if args.exp_starts_with is not None:
    configs = configs[configs.exp_name.str.startswith(args.exp_starts_with)]

assert(len(configs.dataset.unique()) == 1)
dataset = configs.dataset.iloc[0]

if args.debug:
    configs = configs.iloc[:100]

if dataset in ['MIMIC', 'CXP']: 
    group_vars = ['sex', 'ethnicity', 'age'] 
else:
    group_vars =  ['sex']

val_cohorts = {
   str(i): data.load_df(dataset, str(i))['val'] for i in range(5)
}
test_cohort = data.load_df(dataset, str(0))['test']

test_df_raw = []
thress = []
for i, row in configs.iterrows():    
    result_dir = Path(row['output_dir'])
    assert (result_dir/'done').is_file()
    assert (result_dir/'results.pkl').is_file()

    if (result_dir/'args.json').is_file():
        hparams = json.load((result_dir/'args.json').open('r'))
    else:
        hparams = torch.load(result_dir/'results.pkl')['hparams']
    
    res = torch.load(result_dir/'results.pkl')

    pred_df = res['test_pred_df']

    pred_df = pred_df.rename(columns = {
        col : 'pred_' + col
        for col in pred_df if col not in ['path', 'subject_id'] and not col.startswith('pred')
    })

    if hparams['task'] == 'multitask':
        if row['exp_name'] == args.baseline_exp_name:
            thress.append(res['test_metrics']['optimal_thres'])
        target_cols = Constants.take_labels
    else:
        if row['exp_name'] == args.baseline_exp_name:
            thress.append({hparams['task']: res['test_metrics']['optimal_thres']})
        target_cols = [hparams['task']]
        if 'pred' in pred_df:
            pred_df = pred_df.rename(columns = {'pred': 'pred_' + hparams['task']})
        
    pred_cols = ['pred_' + i for i in target_cols]
    test_df_i = test_cohort[['path'] + group_vars + target_cols].merge(pred_df, on = 'path', how = 'left')  
        
    assert(test_df_i[target_cols[0]].isnull().sum() == 0)
    for r in configs.columns:
        test_df_i[r] = row[r]
    test_df_raw.append(test_df_i)

    # postprocessing
    if row['exp_name'] == args.baseline_exp_name and len(args.postprocess_thresholds):
        # load validation predictions
        val_df_i = res['val_pred_df'].rename(columns = {
            col : 'pred_' + col
            for col in res['val_pred_df'] if col not in ['path', 'subject_id'] and not col.startswith('pred')
        })
        if 'pred' in val_df_i:
            val_df_i = val_df_i.rename(columns = {'pred': 'pred_' + hparams['task']})
        val_df_i = val_cohorts[hparams['val_fold']][['path'] + group_vars + target_cols].merge(val_df_i, on = 'path', how = 'right')      
        assert(val_df_i[target_cols[0]].isnull().sum() == 0)

        for thres_post in args.postprocess_thresholds:
            for grp in group_vars:
                test_df_i_post = test_df_i.copy()
                for target_col, pred_col in zip(target_cols, pred_cols):
                    eo = CalibratedEqualizedOddsPostProcessor(cost_weights={ "fpr": 1,"fnr": 1/thres_post - 1 }, thresholds = [thres_post])
                    eo(torch.tensor(val_df_i[pred_col]).float(), torch.tensor(val_df_i[target_col]).int(), val_df_i[grp].values)
                    test_df_i_post[pred_col], _ = eo.postprocess(torch.tensor(test_df_i_post[pred_col]).float(), test_df_i_post[grp].values)

                test_df_i_post['protected_attr'] = grp
                test_df_i_post['exp_name'] = row['exp_name']+'_post_' + str(thres_post)
                test_df_raw.append(test_df_i_post)
    
test_df = pd.concat(test_df_raw)
thress = pd.DataFrame(thress).mean(axis = 0)

raws = []
for grp in group_vars:
    for target_name, pred_name in zip(target_cols, pred_cols):                
        sub_df = test_df[(test_df.protected_attr == grp) | pd.isnull(test_df.protected_attr)]
        sub_df = sub_df[~pd.isnull(sub_df[pred_name])]
        
        evaluator = StandardEvaluator(thresholds = args.thresholds + ([thress[target_name]] if args.baseline_exp_name is not None else []),
                                sens_at_spec = args.sens_at_spec)
        temp = (evaluator.bootstrap_evaluate(sub_df, n_boot = args.n_boot,
                                            strata_vars_eval = ["exp_name", "val_fold", grp],
                                            strata_vars_boot=[grp],
                                            strata_var_replicate="val_fold",
                                            replicate_aggregation_mode=None,
                                            strata_var_experiment="exp_name",
                                            strata_var_group=grp,
                                            baseline_experiment_name=args.baseline_exp_name,
                                            compute_overall=True,
                                            compute_group_min_max=True,
                                            label_var = target_name, 
                                            pred_prob_var = pred_name,
                                            patient_id_var="path",
                                            n_jobs = args.n_jobs)
                .rename(columns = {grp: 'grp_val'}))
        
        temp['target'] = target_name
        temp['eval_group'] = grp
        raws.append(temp)                    
        
df_all = pd.concat(raws, ignore_index = True)       
df_all.to_pickle(args.out_path)
