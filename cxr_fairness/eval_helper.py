import torch
import numpy as np
from cxr_fairness.lib import misc
from sklearn.metrics import roc_auc_score
import pandas as pd
from cxr_fairness.data import Constants
from cxr_fairness.utils import compute_opt_thres

def predict_on_set(algorithm, loader, device, add_fields = ('sex', 'race')):
    preds, targets, paths, adds = [], [], [], {i:[] for i in add_fields}
    with torch.no_grad():
        for x, y, meta in loader:
            x = misc.to_device(x, device)
            algorithm.eval()
            logits = algorithm.predict(x)

            targets += y.detach().cpu().numpy().tolist()
            paths += meta['path']
            for j in add_fields:
                adds[j] += meta[j]

            if y.ndim == 1 or y.shape[1] == 1: # multiclass
                preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
            else: # multilabel
                preds_list = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            if isinstance(preds_list, list):
                preds += preds_list
            else:
                preds += [preds_list]
    return np.array(preds), np.array(targets), np.array(paths), adds

def eval_metrics(algorithm, loader, device, protected_attr = None):
    preds, targets, paths, adds = predict_on_set(algorithm, loader, device, 
        add_fields = ((protected_attr,) if protected_attr is not None else ())
    )    
    # mean AUC
    if targets.ndim == 2: # multitask
        mean_auc = np.mean([roc_auc_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])])
        pred_df = pd.DataFrame({**{'path': paths}, **{label: preds[:, c] for c, label in enumerate(Constants.take_labels)}})
        optimal_thress = {label: compute_opt_thres(targets[:, c], preds[:, c]) for c, label in enumerate(Constants.take_labels)}
    else:
        mean_auc = roc_auc_score(targets, preds)
        pred_df = pd.DataFrame({'path': paths, 'pred': preds})
        optimal_thress = compute_opt_thres(targets, preds)
     
    # compute worst AUC
    aucs = []
    if protected_attr is not None:
        adds[protected_attr] = np.array(adds[protected_attr])
        unique_groups = np.unique(adds[protected_attr])
        for grp in unique_groups:
            mask = adds[protected_attr] == grp
            if targets.ndim == 2:
                aucs.append(np.mean([roc_auc_score(targets[:, i][mask], preds[:, i][mask]) for i in range(targets.shape[1])])) 
            else:
                aucs.append(roc_auc_score(targets[mask], preds[mask])) 

    return {
        'roc': mean_auc,
        'worst_roc': mean_auc if not len(aucs) else min(aucs),
        'roc_gap': 0. if not len(aucs) else max(aucs) - min(aucs),
        'optimal_thres': optimal_thress
    }, pred_df
   

