from sklearn.calibration import calibration_curve
from netcal.metrics import ECE   
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
import random

def binary_metrics(targets, preds):    
    if len(targets) == 0:
        return {}
    res = {'accuracy': accuracy_score(targets, preds)}
    CM = confusion_matrix(targets, preds, labels = [0, 1])

    res['n_samples'] = len(targets)

    res['TN'] = CM[0][0].item()
    res['FN'] = CM[1][0].item()
    res['TP'] = CM[1][1].item()
    res['FP'] = CM[0][1].item()

    res['error'] = res['FN'] + res['FP']

    if res['TP']+res['FN'] == 0:
        res['TPR'] = 0
        res['FNR'] = 1
    else:
        res['TPR'] = res['TP']/(res['TP']+res['FN'])
        res['FNR'] = res['FN']/(res['TP']+res['FN'])

    if res['FP']+res['TN'] == 0:
        res['FPR'] = 1
        res['TNR'] = 0
    else:
        res['FPR'] = res['FP']/(res['FP']+res['TN'])
        res['TNR'] = res['TN']/(res['FP']+res['TN'])

    res['precision'] = res['TP'] / (res['TP'] + res['FP']) if (res['TP'] + res['FP']) > 0 else 0
    res['pred_prevalence'] = (res['TP'] + res['FP'])/res['n_samples']    
    res['prevalence'] = (res['TP'] + res['FN'])/res['n_samples']
    res['F1'] = f1_score(targets, preds, zero_division = 0)  
    
    return res

def CIs(row):
    return {
        'mean': row.mean(),
        'lower': np.quantile(row, 0.025),
        'upper': np.quantile(row, 0.975)
    }

def collapse_bootstrap_df(df): 
    temp_df_raw = df.groupby(level = 0).agg({i: CIs for i in df.columns}).to_dict()
    df = pd.DataFrame.from_dict(temp_df_raw, orient="index").stack().to_frame()
    return pd.DataFrame(df[0].values.tolist(), index=df.index)

def get_bootstrapped_metrics(df, attrs, n_boot, pred_col = 'pred', target_col = 'No Finding', seed = 42):
    result_df_raw = []
    st0 = np.random.get_state()
    st1 = random.getstate()
    np.random.seed(seed)
    random.seed(seed)     

    for boot in range(n_boot):
        rows = {}                
        for attr in attrs:
            assert attr in df.columns, attr
            sampled_df = stratify_shuffle(df, stratify_by = attr)           
            for grp in sampled_df[attr].unique():
                mask = sampled_df[attr] == grp
                rows[grp] = binary_metrics(sampled_df.loc[mask, target_col], sampled_df.loc[mask, pred_col])
        result_df_raw.append(pd.DataFrame(rows).T)
    np.random.set_state(st0)
    random.setstate(st1)

    return collapse_bootstrap_df(pd.concat(result_df_raw))

def stratify_shuffle(df, stratify_by):
    if stratify_by is None:
        idxs = np.random.choice(np.arange(len(df)), size = len(df))
        return df.iloc[idxs]
    else:
        return df.groupby(stratify_by).apply(lambda x: x.sample(n = len(x), replace = True)).sample(frac=1).reset_index(drop = True)

def group_bin_metrics(targets, preds, group, thres):
    res = {}
    for c, grp in enumerate(np.unique(group)):
        mask = group == grp
        res[grp] = binary_metrics(targets[mask], (preds >= thres)[mask])
    return pd.DataFrame(res)

def plot_auc_curve(targets, preds, mask, label, compr_thres, color):
    fpr, tpr, thresholds = roc_curve(targets[mask], preds[mask])
    plt.plot(fpr, tpr, label = label, c = color)
    if compr_thres is not None:
        ind_pt = np.argmin(np.abs(thresholds - compr_thres))
        plt.plot(fpr[ind_pt], tpr[ind_pt], 'o', markersize = 10, c = color)
    ax = plt.gca()
    ax.set_ylabel('TPR')
    ax.set_xlabel("FPR")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
        
def plot_all_auc_curves(targets, preds, group, compr_thres):
    plt.figure()
    aucs = {}
    for c, grp in enumerate(np.unique(group)):
        mask = group == grp
        plot_auc_curve(targets, preds, mask, label = grp, compr_thres = compr_thres, color = 'C'+str(c))
        aucs[grp] = roc_auc_score(targets[mask], preds[mask])
    plt.legend()
    return aucs

def plot_calibration_curves(targets, preds, group, n_bins = 10):
    plt.figure()
    eces = {}
    for c, grp in enumerate(np.unique(group)):
        mask = group == grp
        prob_t, prob_p = calibration_curve(targets[mask], preds[mask], n_bins = n_bins)
        plt.plot(prob_p, prob_t, label = grp, c = 'C' + str(c))
        eces[grp] = ECE().measure(y= targets[mask].values, X= preds[mask].values)
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_ylabel('Fraction of Positives')
    ax.set_xlabel("Predicted Value")
    ax.plot([0, 1], [0, 1], "k:")
    eces['All'] = ECE().measure(y= targets.values, X= preds.values)
    return eces

def plot_risk_curves(targets, preds, groups):
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5), dpi = 400)
    ax = axs[0]
    for grp in np.unique(groups):
        X = preds[groups == grp]
        ax.hist(X, bins = 50, edgecolor='black', linewidth=1.2, alpha = 0.3, density = True)
    ax.set_xlim(left = 0, right = 1)
    ax.set_title('$P(h_{\\theta} | G = G_k)$')
    
    for ax, tval in zip([axs[1], axs[2]], [0, 1]):
        for grp in sorted(np.unique(groups)):
            X = preds[(groups == grp) & (targets == tval)]
            ax.hist(X, bins = 50, edgecolor='black', linewidth=1.2, alpha = 0.3, density = True)
        ax.set_xlim(left = 0, right = 1)
        ax.set_title('$P(h_{\\theta} | G = G_k, Y = %s)$'% tval)
        
    legend = [
        mlines.Line2D([], [], linestyle = '-', color = 'C'+str(c), marker ='.',
                     label = eth) for c, eth in enumerate(sorted(np.unique(groups)))
    ]

    leg1 = axs[-1].legend(handles = legend, loc='center left', bbox_to_anchor=(1.1,0.5))  
    
    return fig