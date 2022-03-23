import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import socket
import itertools

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader

from cxr_fairness import algorithms
from cxr_fairness.lib import misc
from cxr_fairness.data import Constants, data
from cxr_fairness.lib.infinite_data_loader import InfiniteDataLoader
from cxr_fairness.utils import EarlyStopping, has_checkpoint, load_checkpoint, save_checkpoint
from cxr_fairness import eval_helper

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')
parser.add_argument('--val_fold', type=str, choices = list(map(str, range(5))), required = True)
parser.add_argument('--dataset', type=str, choices = ['MIMIC', 'CXP'], required = True)
parser.add_argument('--task', type=str, choices = ['multitask'] + Constants.take_labels_all, required = True)
parser.add_argument('--data_type', type=str, choices = ['normal', 'balanced', 'single_group'], required = True)
parser.add_argument('--model', type=str, choices = ['densenet', 'resnet', 'vision_transformer'], default = 'densenet')
parser.add_argument('--algorithm', type=str, choices=["ERM", 'GroupDRO', 'DistMatch', 'SimpleAdv', 'ARL', 'FairALM', 'JTT'], default = 'ERM')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--use_cache', action = 'store_true')
parser.add_argument('--concat_group', action = 'store_true')
parser.add_argument('--only_frontal', action = 'store_true')
parser.add_argument('--smaller_label_set', action = 'store_true', help = 'use labels only available across all datasets')
parser.add_argument('--protected_attr', type=str, choices = ['sex', 'ethnicity', 'age'])
parser.add_argument('--subset_group', type = str, choices = [str(i) for i in itertools.chain(*Constants.group_vals.values())])
parser.add_argument('--val_subset', type = int, default = None, help = 'use a subset of the validation set for early stopping')
parser.add_argument('--checkpoint_freq', type = int, default = 200)
parser.add_argument('--es_patience', type = int, default = 5) # * checkpoint_freq steps
parser.add_argument('--max_steps', type = int, default = 20000) 
parser.add_argument('--delete_model', action = 'store_true', 
    help = 'delete model weights after training to save disk space')
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--es_metric', type = str, choices = ['roc', 'worst_roc'], default = 'worst_roc')
parser.add_argument('--weight_decay', type = float, default = 0.0)
parser.add_argument('--clf_head_ratio', type = float, default = 2.0, 
    help = 'ratio between layer output sizes in 2-layer clf head')
parser.add_argument('--debug', action = 'store_true')
# Group DRO
parser.add_argument('--groupdro_eta', type = float, default = 0.1)
# DistMatch
parser.add_argument('--distmatch_penalty_weight', type = float, default = 0.1)
parser.add_argument('--match_type', choices = ['MMD', 'mean'], default = 'mean')
# SimpleAdv
parser.add_argument('--adv_alpha', type = float, default = 0.1)
# FairALM
parser.add_argument('--fairalm_threshold', type = float, default = 0.5)
parser.add_argument('--fairalm_surrogate', type = str, default = 'logistic', choices = ['logistic', 'sigmoid', 'hinge'])
parser.add_argument('--fairalm_eta', type = float, default = 1e-3)
# JTT
parser.add_argument('--JTT_ERM_model_folder', type = str)
parser.add_argument('--JTT_weight', default = 3.0, type = float)
parser.add_argument('--JTT_threshold', default = 0.5, type = float)
args = parser.parse_args()

if args.subset_group in  ['0', '1', '2']:
    args.subset_group = int(args.subset_group)

if args.debug:
    args.val_subset = 512
    args.max_steps = 100

hparams = vars(args)
hparams['num_classes'] = len(Constants.take_labels) if hparams['task'] == 'multitask' else 2
os.makedirs(args.output_dir, exist_ok=True)
sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tTorchvision: {}".format(torchvision.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tPIL: {}".format(PIL.__version__))
print("\tNode: {}".format(socket.gethostname()))

print('Args:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

with open(os.path.join(args.output_dir,'args.json'), 'w') as f:
    json.dump(hparams, f, indent = 4)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if hparams['algorithm'] in ['GroupDRO',  'MMD', 'MeanMatch', 'SimpleAdv', 'FairALM']:
    assert hparams['data_type'] == 'balanced'

if hparams['data_type'] == 'single_group':
    assert hparams['subset_group'] in Constants.group_vals[hparams['protected_attr']]
    query_str = f"{hparams['protected_attr']} == '{hparams['subset_group']}'"    
else:
    query_str = None

dfs_all = data.load_df(hparams['dataset'], hparams['val_fold'], query_str = query_str, only_frontal = args.only_frontal)

ds_args = {
    'env' : hparams['dataset'],
    'concat_group': hparams['concat_group'],
    'protected_attr': hparams['protected_attr'], 
    'imagenet_norm': True,
    'use_cache': hparams['use_cache'], 
    'smaller_label_set': hparams['smaller_label_set'],
    'subset_label': None if hparams['task'] == 'multitask' else hparams['task']
}

if hparams['data_type'] in ['normal', 'single_group']:
    train_dss = [data.get_dataset(dfs_all, split = 'train', augment = 1, **ds_args)]
else:
    query_strs = [f'{args.protected_attr}=="{group}"' for group in Constants.group_vals[args.protected_attr]]
    train_dss = [data.get_dataset(data.load_df(hparams['dataset'], hparams['val_fold'], query_str = query_str), 
                        split = 'train', augment = 1, **ds_args) for query_str in query_strs]

train_loaders = [InfiniteDataLoader(
        dataset=i,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=1)
        for i in train_dss
        ]
val_ds = data.get_dataset(dfs_all, split = 'val', augment = 0, **ds_args)
test_ds = data.get_dataset(dfs_all, split = 'test', augment = 0, **ds_args)

if args.val_subset:
    val_ds_es = torch.utils.data.Subset(val_ds, np.random.choice(np.arange(len(val_ds)), min(args.val_subset, len(val_ds)), replace = False))
else:
    val_ds_es = val_ds

if args.debug:
    test_ds = torch.utils.data.Subset(test_ds, list(range(512)))

eval_loader = DataLoader(
        dataset=val_ds_es,
        batch_size=hparams['batch_size']*4,
        num_workers=1)
    
test_loader = DataLoader(
        dataset=test_ds,
        batch_size=hparams['batch_size']*4,
        num_workers=1)

algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(hparams['num_classes'], hparams)
algorithm.to(device)

print("Number of parameters: %s" % sum([np.prod(p.size()) for p in algorithm.parameters()]))

train_minibatches_iterator = zip(*train_loaders)   
steps_per_epoch = min([len(i)/hparams['batch_size'] for i in train_dss])
n_steps = args.max_steps
checkpoint_freq = args.checkpoint_freq
es = EarlyStopping(patience = args.es_patience)

if has_checkpoint():
    state = load_checkpoint()
    algorithm.load_state_dict(state['model_dict'])
    algorithm.optimizer.load_state_dict(state['optimizer_dict'])
    [train_loader.sampler.load_state_dict(state['sampler_dicts'][c]) for c, train_loader in enumerate(train_loaders)]
    start_step = state['start_step']
    es = state['es']
    torch.random.set_rng_state(state['rng'])
    print("Loaded checkpoint at step %s" % start_step)
else:
    start_step = 0        

checkpoint_vals = collections.defaultdict(lambda: [])
last_results_keys = None
for step in range(start_step, n_steps):
    if es.early_stop:
        break
    step_start_time = time.time()
    minibatches_device = [(misc.to_device(xy[0], device), misc.to_device(xy[1], device))
        for xy in next(train_minibatches_iterator)]
    algorithm.train()
    step_vals = algorithm.update(minibatches_device, device)
    checkpoint_vals['step_time'].append(time.time() - step_start_time)

    for key, val in step_vals.items():
        checkpoint_vals[key].append(val)
        
    if step % checkpoint_freq == 0:
        results = {
            'step': step,
            'epoch': step / steps_per_epoch,
        }

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        results.update(eval_helper.eval_metrics(algorithm, eval_loader, device = device, protected_attr = args.protected_attr)[0])
            
        results_keys = sorted(results.keys())
        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys],
            colwidth=12)

        results.update({
            'hparams': hparams,   
        })

        epochs_path = os.path.join(args.output_dir, 'results.jsonl')
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")
        
        save_checkpoint(algorithm, algorithm.optimizer, 
                        [train_loader.sampler.state_dict(train_loader._infinite_iterator) for c, train_loader in enumerate(train_loaders)], 
                        step+1, es, torch.random.get_rng_state())
        
        checkpoint_vals = collections.defaultdict(lambda: [])
        
        es(-results[args.es_metric], step, algorithm.state_dict(), os.path.join(args.output_dir, "model.pkl"))            

algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pkl")))
algorithm.eval()

save_dict = {
    "hparams": hparams,
    "es_step": es.step,
    'es_roc': es.best_score
}

val_loader = DataLoader(
        dataset=val_ds,
        batch_size=hparams['batch_size']*4,
        num_workers=1)

val_metrics, val_pred_df = eval_helper.eval_metrics(algorithm, val_loader, device = device, protected_attr = args.protected_attr)
test_metrics, test_pred_df = eval_helper.eval_metrics(algorithm, test_loader, device = device, protected_attr = args.protected_attr)
    
save_dict['val_metrics'] = val_metrics
save_dict['val_pred_df'] = val_pred_df    
save_dict['test_metrics'] = test_metrics
save_dict['test_pred_df'] = test_pred_df

torch.save(save_dict, os.path.join(args.output_dir, "results.pkl"))    

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')

if args.delete_model:
    os.remove(os.path.join(args.output_dir, "model.pkl"))