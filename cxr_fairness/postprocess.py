from os import fpathconf
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import operator
import warnings
from collections import OrderedDict

## Code here adapted from: https://github.com/fairlearn/fairlearn/tree/main/fairlearn/postprocessing

def get_postprocess_class(postprocess_name):
    """Return the post-processing class with the given name."""
    if postprocess_name not in globals():
        raise NotImplementedError("Post-processing algorithm not found: {}".format(postprocess_name))
    return globals()[postprocess_name]

def _get_confusion_matrix(tp, fp, tn, fn):
    return_dict = {"true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                    "predicted_positives": tp+fp,
                    "predicted_negatives": tn+fn,
                    "positives": tp+fn,
                    "negatives": tn+fp,
                    "n": tp+fp+tn+fn
    }
    return return_dict

def _fpr(x):
    return x["false_positives"] / x["negatives"]
    
def _tnr(x):
    return x["true_positives"] / x["negatives"]

def _tpr(x):
    return x["true_positives"] / x["positives"]
    
def _fnr(x):
    return x["false_negatives"] / x["positives"]
    
def _accuracy_score(x):
    return (x["true_positives"] + x["true_negatives"]) / x["n"]
    
def _balanced_accuracy_score(x):
    return 0.5 * x["true_positives"] / x["positives"] + 0.5 * x["true_negatives"] / x["negatives"]
    
def _get_convex_hull(points):
    selected = []
    listlen = len(points["x"])
    assert listlen == len(points["y"]) and listlen == len(points["threshold"])
    
    for i in range(listlen):
        r2 = {"x": points["x"][i], "y": points["y"][i], "threshold": points["threshold"][i]}
        while len(selected) >= 2:
            r1 = selected[-1]
            r0 = selected[-2]
            
            if (r1["y"] - r0["y"])*(r2["x"] - r0["x"]) <= (r2["y"] - r0["y"])*(r1["x"] - r0["x"]):
                selected.pop()
            else:
                break
        selected.append(r2)
    return selected

### NOTE: Both EqualizedOdds and CalibratedEqualizedOdds PostProcessors rely on the assumption that 
### the classification task reduces to the binary case (i.e. the task is binary classification, or,
### if there are greater than two labels, the task is multi-label, and can therefore be assessed as 
### separate binary classification tasks over each label for the purpose of post-processing)
class EqualizedOddsPostProcessor:
    def __init__(self, grid_size=1000, objective="accuracy_score", binary=True, random_state=1):
        self.name = "EqualizedOddsPostProcessor"
        self.grid_size = grid_size
        self.objective = objective
        self.binary = binary
        self.random_state = random_state
        
        if self.objective == "accuracy_score":
            self.objective = _accuracy_score
        elif self.objective == "balanced_accuracy_score":
            self.objective = _balanced_accuracy_score
        else:
            raise ValueError("Objective for EO Post-processing must be in ['accuracy_score', 'balanced_accuracy_score'], {} cannot be used".format(self.objective))
            
        self.postprocessed = False
        
    def _tradeoff_curve(self, probs, targets, unique_group):
    
        sorted_idx = probs.argsort()[::-1]
        sorted_probs = list(probs[sorted_idx])
        sorted_targets = list(targets[sorted_idx])
        n = len(targets)
        n_positive = targets.sum()
        n_negative = n - n_positive
        
        assert n_positive > 0 and n_negative > 0, "Degenerate group {} has only 1 label".format(unique_group)
        
        sorted_probs.append(-np.inf)
        sorted_targets.append(np.nan)
        
        i = 0
        count = [0,0]
        x_list, y_list, threshold_list = [], [], []
        while i < n:
            if not len(x_list):
                threshold = np.inf
            else:
                threshold = sorted_probs[i]
                while sorted_probs[i] == threshold:
                    count[int(sorted_targets[i])] += 1
                    i += 1
                threshold = (threshold + sorted_probs[i]) / 2
            
            actual_counts = _get_confusion_matrix(tp = count[1], fp = count[0], tn = (n_negative - count[0]), fn = (n_positive-count[1]))
            
            x_list.append(_fpr(actual_counts))
            y_list.append(_tpr(actual_counts))
            threshold_list.append(threshold)
        
        sorted_lists = sorted(zip(x_list, y_list, list(range(len(x_list)))))
        sorted_x_list = [x for x, _, _ in sorted_lists]
        sorted_y_list = [y for _, y, _ in sorted_lists]
        sorted_threshold_list = [threshold_list[i] for _, _, i in sorted_lists]
        sorted_points =  {"x": sorted_x_list, "y": sorted_y_list, "threshold": sorted_threshold_list}
        
        selected_points = _get_convex_hull(sorted_points)
        
        return selected_points
    
    def _interpolate_curve(self, convex_hull_list, _x_grid):
        i = 0
        dict_list = []
        x0 = convex_hull_list[0]["x"]
        while convex_hull_list[i+1]["x"] == x0:
            i += 1
        
        for x in _x_grid:
            while x > convex_hull_list[i+1]["x"]:
                i += 1
            
            x_distance_from_next_data_point = convex_hull_list[i+1]["x"] - x
            x_distance_between_data_points = convex_hull_list[i+1]["x"] - convex_hull_list[i]["x"]
            p0 = x_distance_from_next_data_point/x_distance_between_data_points
            p1 = 1 - p0
            y = p0 * convex_hull_list[i]["y"] + p1 * convex_hull_list[i+1]["y"]
            dict_list.append({
                            "x": x,
                            "y": y,
                            "p0": p0,
                            "threshold0": convex_hull_list[i]["threshold"],
                            "p1": p1,
                            "threshold1": convex_hull_list[i+1]["threshold"]
                             })
        
        return dict_list
        
    def optimize_threshold(self, probs, targets, groups):
        n = len(targets)
        n_positive = targets.sum()
        n_negative = n - n_positive
        
        _tradeoff_curve = {}
        _x_grid = np.linspace(0,1,self.grid_size+1)
        y_values = {}
        
        for unique_group in np.unique(groups):
            grp_targets = targets[groups == unique_group]
            grp_probs = probs[groups == unique_group]
            
            roc_convex_hull = self._tradeoff_curve(grp_probs, grp_targets, unique_group)
            
            _tradeoff_curve[unique_group] = self._interpolate_curve(roc_convex_hull, _x_grid)
            y_values[unique_group] = [point["y"] for point in _tradeoff_curve[unique_group]]
        
        _y_min = np.amin(np.stack([y_values[key] for key in y_values]).T, axis=1)
        
        counts = _get_confusion_matrix(tp = (n_positive * _y_min), fp = (n_negative * _x_grid), tn = (n_negative * (1.0 - _x_grid)), fn = (n_positive *(1.0 - _y_min)))
        objective_values = np.around(self.objective(counts), 15)

        idx_best = np.argmax(objective_values)

        _x_best = _x_grid[idx_best]
        _y_best = _y_min[idx_best]
        
        interpolation_dict = {}
        for unique_group in _tradeoff_curve.keys():
            roc_result = _tradeoff_curve[unique_group][idx_best]

            if roc_result["x"] == roc_result["y"]:
                p_ignore = 0
            else:
                difference_from_best_predictor_for_group = (roc_result["y"] - _y_best)
                vertical_distance_from_diagonal = roc_result["y"] - roc_result["x"]
                p_ignore = (difference_from_best_predictor_for_group / vertical_distance_from_diagonal)
            interpolation_dict[unique_group] = {
                "p_ignore": p_ignore, 
                "prediction_constant": _x_best, 
                "p0": roc_result["p0"], 
                "threshold0": roc_result["threshold0"], 
                "p1": roc_result["p1"], 
                "threshold1": roc_result["threshold1"]
                }
        
        return interpolation_dict
        
    def __call__(self, logits, targets, groups):
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits).float()
        probs = F.sigmoid(logits)
        
        if torch.is_tensor(targets): targets = targets.detach().cpu().numpy()
        if torch.is_tensor(probs): probs = probs.detach().cpu().numpy()
        if self.binary:
            interpolation_dict_list = self.optimize_threshold(probs, targets, groups)
        else:
            unique_targets = np.unique(targets)
            interpolation_dict_list = {}
            for unique_target in unique_targets:
                binary_probs = probs[:,unique_target]
                binary_targets = targets == unique_target
                per_target_interpolation_dict = self.optimize_threshold(binary_probs, binary_targets, groups)
                interpolation_dict_list[unique_target] = per_target_interpolation_dict
        self.interpolation_dict_list = interpolation_dict_list
        self.postprocessed = True
        
    def postprocess(self, logits, groups):
        assert self.postprocessed and hasattr(self, "interpolation_dict_list"), "{} has not yet been called! Must be called before 'postprocess' can be called on this object.".format(self.name)
        
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits).float()
        
        probs = F.sigmoid(logits)
        if self.binary:
            positive_probs = 0.0*probs
            for unique_group, interpolation_dict in self.interpolation_dict_list.items():
                interpolated_predictions = interpolation_dict["p0"]*(probs > interpolation_dict["threshold0"])+interpolation_dict["p1"]*(probs > interpolation_dict["threshold1"])
                positive_probs[groups == unique_group] = interpolated_predictions[groups == unique_group]
        else:
            positive_probs_dict = {}
            for unique_target, per_target_interpolation_dict in self.interpolation_dict_list.items():
                per_target_probs = probs[:, unique_target]
                per_target_positive_probs = 0.0*per_target_probs
                for unique_group, interpolation_dict in per_target_interpolation_dict.items():
                    interpolated_predictions = interpolation_dict["p0"]*(per_target_probs > interpolation_dict["threshold0"])+interpolation_dict["p1"]*(per_target_probs > interpolation_dict["threshold1"])
                    if "p_ignore" in interpolation_dict:
                        interpolated_predictions = interpolation_dict["p_ignore"]*interpolation_dict["prediction_constant"] + (1-interpolation_dict["p_ignore"])*interpolated_predictions
                    per_target_positive_probs[groups == unique_group] = interpolated_predictions[groups == unique_group]
                positive_probs_dict[unique_target] = per_target_positive_probs
            positive_probs_dict = OrderedDict(sorted(positive_probs_dict.items()))
            positive_probs = torch.stack([values for _, values in positive_probs_dict.items()]).T
        
        torch.manual_seed(self.random_state)
        predictions = (positive_probs >= torch.rand(positive_probs.shape, device=positive_probs.device)).long()
        return positive_probs, predictions
                

class CalibratedEqualizedOddsPostProcessor:
    def __init__(self, cost_weights={"fpr": 1., "fnr": 1.}, thresholds=[0.5], binary=True):
        self.name = "CalibratedEqualizedOddsPostProcessor"
        self.cost_weights = cost_weights
        self.thresholds = thresholds
        self.binary = binary

        assert ("fpr" in self.cost_weights and "fnr" in self.cost_weights) or \
                ("fpr" in self.cost_weights.values() and "fnr" in self.cost_weights.values()), \
                    "Cost weights must be a dictionary object or nested dictionary with both fpr and fnr as keys in the innermost dictionary."

        self.postprocessed = False
        
    def _base_rate(self, targets):
        return np.mean(targets)
        
    def _trivial(self, targets):
        base_rate = self._base_rate(targets)
        predictions = np.ones(targets.shape)*base_rate
        return predictions
    
    def _weighted_cost(self, counts, targets, cost_weights):
        fpr = _fpr(counts)
        fnr = _fnr(counts)
        fp_weight = cost_weights["fpr"]
        fn_weight = cost_weights["fnr"]
        base_rate = self._base_rate(targets)
        
        normalizing_const = (fp_weight + fn_weight) if fp_weight != 0 and fn_weight != 0 else 1
        weighted_cost = (fp_weight * fpr * (1-base_rate) + fn_weight * fnr * base_rate) / normalizing_const
        return weighted_cost

    def find_interpolation(self, probs, targets, groups, threshold):
        unique_groups = np.unique(groups)
        group_costs = {}
        trivial_group_costs = {}
        for unique_group in unique_groups:
            if unique_group in self.cost_weights:
                cost_weights = self.cost_weights[unique_group]
            else:
                cost_weights = self.cost_weights
            ### Calculate cost for model predictions
            grp_targets = targets[groups == unique_group]
            grp_probs = probs[groups == unique_group]
            
            grp_predictions = (grp_probs >= threshold).astype(int)

            tp = np.sum(np.logical_and(grp_predictions == 1, grp_targets == 1))
            fp = np.sum(np.logical_and(grp_predictions == 1, grp_targets == 0))
            fn = np.sum(np.logical_and(grp_predictions == 0, grp_targets == 1))
            tn = np.sum(np.logical_and(grp_predictions == 0, grp_targets == 0))

            grp_counts = _get_confusion_matrix(tp, fp, tn, fn)
            group_costs[unique_group] = self._weighted_cost(grp_counts, grp_targets, cost_weights)

            ### Calculate cost for the trivial model predictions
            trivial_grp_probs = self._trivial(grp_targets)
            trivial_grp_predictions = (trivial_grp_probs >= threshold).astype(int)

            tp = np.sum(np.logical_and(trivial_grp_predictions == 1, grp_targets == 1))
            fp = np.sum(np.logical_and(trivial_grp_predictions == 1, grp_targets == 0))
            fn = np.sum(np.logical_and(trivial_grp_predictions == 0, grp_targets == 1))
            tn = np.sum(np.logical_and(trivial_grp_predictions == 0, grp_targets == 0))

            trivial_grp_counts = _get_confusion_matrix(tp, fp, fn, tn)
            trivial_group_costs[unique_group] = self._weighted_cost(trivial_grp_counts, grp_targets, cost_weights)
        
        ### Find group with maximum cost
        max_grp, max_cost = max(group_costs.items(), key=operator.itemgetter(1))

        ### If not all other groups have larger trivial costs than max group, raise warning
        if not np.all(np.array(list(trivial_group_costs.values())) >= max_cost):
            warnings.warn("There exist groups that have lower trivial cost than the maximum group cost for the given model. In these cases, the constraint will not be satisfied.")

        interpolation_dict = {}
        for unique_group in unique_groups:
            grp_targets = targets[groups == unique_group]
            grp_base_rate = self._base_rate(grp_targets)
            if unique_group == max_grp or trivial_group_costs[unique_group] < max_cost:
                p = 0
            else:
                p = (max_cost-group_costs[unique_group]) / (trivial_group_costs[unique_group] - group_costs[unique_group])
            interpolation_dict[unique_group] = {"base_rate": grp_base_rate, "p": p}
        return interpolation_dict
            
    def __call__(self, logits, targets, groups):
        probs = F.sigmoid(logits)
        
        if torch.is_tensor(targets): targets = targets.detach().cpu().numpy()
        if torch.is_tensor(probs): probs = probs.detach().cpu().numpy()
        if self.binary:
            interpolation_dict_list = self.find_interpolation(probs, targets, groups, self.thresholds[0])
        else:
            unique_targets = np.unique(targets)
            if len(self.thresholds) == 1:
                self.thresholds = np.array(self.thresholds*len(unique_targets))

            assert len(self.thresholds) == len(unique_targets), "Number of unique provided targets ({}) does not match the length of provided thresholds ({})" \
                .format(len(self.thresholds),len(unique_targets))

            interpolation_dict_list = {}
            for unique_target in unique_targets:
                binary_probs = probs[:,unique_target]
                binary_targets = targets == unique_target
                binary_threshold = self.thresholds[unique_target]
                per_target_interpolation_dict = self.find_interpolation(binary_probs, binary_targets, groups, binary_threshold)
                interpolation_dict_list[unique_target] = per_target_interpolation_dict
        self.interpolation_dict_list = interpolation_dict_list
        self.postprocessed = True
        
    def postprocess(self, logits, groups):
        assert self.postprocessed and hasattr(self, "interpolation_dict_list"), "{} has not yet been called! Must be called before 'postprocess' can be called on this object.".format(self.name)
        
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits).float()
                
        probs = F.sigmoid(logits)
        if self.binary:
            positive_probs = 0.0*probs
            for unique_group, interpolation_dict in self.interpolation_dict_list.items():
                interpolated_predictions = probs[groups == unique_group]
                trivial_indices = np.random.choice(a=[True, False], size=len(interpolated_predictions), p=[interpolation_dict["p"], 1-interpolation_dict["p"]])
                interpolated_predictions[trivial_indices] = interpolation_dict["base_rate"]
                positive_probs[groups == unique_group] = interpolated_predictions
        else:
            positive_probs_dict = {}
            for unique_target, per_target_interpolation_dict in self.interpolation_dict_list.items():
                per_target_probs = probs[:, unique_target]
                per_target_positive_probs = 0.0*per_target_probs
                for unique_group, interpolation_dict in per_target_interpolation_dict.items():
                    interpolated_predictions = per_target_probs[groups == unique_group]
                    trivial_indices = np.random.choice(a=[True, False], size=len(interpolated_predictions), p=[interpolation_dict["p"], 1-interpolation_dict["p"]])
                    interpolated_predictions[trivial_indices] = interpolation_dict["base_rate"]
                    per_target_positive_probs[groups == unique_group] = interpolated_predictions
                positive_probs_dict[unique_target] = per_target_positive_probs
            positive_probs_dict = OrderedDict(sorted(positive_probs_dict.items()))
            positive_probs = torch.stack([values for _, values in positive_probs_dict.items()]).T
        
        if len(self.thresholds) > 1:
            predictions = (positive_probs >= torch.tensor(self.thresholds, device=positive_probs.device).float()).long()
        else:
            predictions = (positive_probs >= self.thresholds[0]).long()
        return positive_probs, predictions