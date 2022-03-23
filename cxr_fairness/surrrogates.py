import numpy as np
import torch
import torch.nn.functional as F

class MetricUndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_surrogate_fn(surrogate_fn):
    if surrogate_fn is None or surrogate_fn == 'logistic':
        return logistic_surrogate
    elif surrogate_fn == "hinge":
        return hinge_surrogate
    elif surrogate_fn == "sigmoid":
        return sigmoid
    else:
        raise ValueError("Surrogate not defined")

def tpr_surrogate(
    outputs, labels, threshold=0.5, surrogate_fn=None
):
    """
        The true positive rate (recall/sensitivity)
    """

    mask = labels == 1
    if mask.sum() == 0:
        raise MetricUndefinedError

    surrogate_fn = get_surrogate_fn(surrogate_fn)

    outputs = F.log_softmax(outputs, dim=1)[:, -1]

    threshold = torch.FloatTensor([threshold]).to(outputs.device)
    threshold = torch.log(threshold)
    return surrogate_fn(outputs[labels == 1] - threshold).mean()

def fpr_surrogate(
    outputs, labels, threshold=0.5, surrogate_fn=None
):
    """
        The false positive rate (1-specificity)
    """
    mask = labels == 0
    if mask.sum() == 0:
        raise MetricUndefinedError

    surrogate_fn = get_surrogate_fn(surrogate_fn)

    outputs = F.log_softmax(outputs, dim=1)[:, -1]

    threshold = torch.FloatTensor([threshold]).to(outputs.device)
    threshold = torch.log(threshold)
    return surrogate_fn(outputs[mask] - threshold).mean()

def sigmoid(x, surrogate_scale=1.0):
    return torch.sigmoid(x * surrogate_scale)

def logistic_surrogate(x):
    # See Bishop PRML equation 7.48
    return torch.nn.functional.softplus(x) / torch.tensor(np.log(2, dtype=np.float32))

def hinge_surrogate(x):
    return torch.nn.functional.relu(1 + x)

def indicator(x):
    return 1.0 * (x > 0)