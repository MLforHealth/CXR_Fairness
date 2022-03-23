import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from itertools import chain
from pathlib import Path
import json

from cxr_fairness import models
from cxr_fairness.data import Constants
from cxr_fairness.surrrogates import tpr_surrogate, fpr_surrogate
from cxr_fairness.utils import grad_reverse

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def cat(lst):
    if torch.is_tensor(lst[0]):
        return torch.cat(lst)
    elif isinstance(lst[0], dict):
        return {i: torch.cat([j[i] for j in lst]) if torch.is_tensor(lst[0][i]) else list(chain([j[i] for j in lst])) for i in lst[0]}

def cross_entropy(logits, y):
    # multiclass
    if y.ndim == 1 or y.shape[1] == 1:
        return F.cross_entropy(logits, y)
    # multitask
    else:
        return F.binary_cross_entropy_with_logits(logits, y.float())


class ERM(nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, num_classes, hparams):
        super().__init__()
        self.hparams = hparams
        self.featurizer = models.get_featurizer(self.hparams)
        self.classifier = models.get_clf_head(hparams, self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )       

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = cross_entropy(self.predict(all_x), all_y.squeeze().long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class GroupDRO(ERM):
    def __init__(self,  num_classes, hparams):
        super(GroupDRO, self).__init__(num_classes, hparams)
        self.register_buffer("q", torch.ones(len(Constants.group_vals[hparams['protected_attr']])))

    def update(self, minibatches, device):     
        assert len(minibatches) == len(self.q)
        if str(self.q.device) != device:
            self.q = self.q.to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}        

class DistMatch(ERM):
    """
    Perform ERM while matching each group's conditional distribution with the marginal, 
        either with MMD or just the mean
    https://arxiv.org/pdf/2007.10306.pdf
    """
    def __init__(self, num_classes, hparams):
        super(DistMatch, self).__init__(num_classes, hparams)

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K/len(gamma)

    def mmd(self, x, y):
        # https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, device):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        all_classifs = torch.cat(classifs)
        targets = [yi for _, yi in minibatches]
        all_targets = torch.cat(targets)

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for target in [0, 1]:
                mask_i = targets[i] == target
                if mask_i.sum() > 0:
                    dist_grp_target = F.log_softmax(classifs[i][mask_i], dim = 1)[:, -1]
                    dist_all_target = F.log_softmax(all_classifs[all_targets == target], dim = 1)[:, -1]
                    if self.hparams['match_type'] == 'MMD':
                        penalty += self.mmd(dist_grp_target.unsqueeze(-1), dist_all_target.unsqueeze(-1))
                    elif self.hparams['match_type'] == 'mean':
                        penalty += (torch.mean(dist_grp_target) -  torch.mean(dist_all_target))**2

        self.optimizer.zero_grad()
        (objective + (self.hparams['distmatch_penalty_weight']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}

class SimpleAdv(ERM):
    '''
    Adversary that predicts the protected group using the label and logits
    https://stanford.edu/~cpiech/bio/papers/fairnessAdversary.pdf
    '''

    def __init__(self, num_classes, hparams):
        super(SimpleAdv, self).__init__(num_classes, hparams)

        self.discriminator = nn.Sequential(
            nn.Linear(3, 3), # 2 logits + y
            nn.ReLU(),
            nn.Linear(3, len(Constants.group_vals[hparams['protected_attr']]))
        )
        
        self.disc_optimizer = torch.optim.Adam( 
            list(self.discriminator.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )  

    def update(self, minibatches, device):
        all_x = cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        classifs = self.classifier(all_z)
        disc_input = grad_reverse(torch.cat((classifs, all_y.unsqueeze(-1)), axis = -1), self.hparams['adv_alpha'])
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((y.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        loss_y = cross_entropy(classifs, all_y.squeeze().long())
        loss_d = cross_entropy(disc_out, disc_labels.squeeze().long())

        loss = loss_y + loss_d

        self.disc_optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        self.disc_optimizer.step()

        return {'loss': loss.item(), 'loss_y': loss_y.item(), 'loss_d': loss_d.item()}

class ARL(ERM):
    '''
    Adversarially reweighted learning
    https://arxiv.org/pdf/2006.13114.pdf
    '''
    def __init__(self, num_classes, hparams):
        super(ARL, self).__init__(num_classes, hparams)

        self.discriminator_emb = models.EmbModel('CBR', pretrain = False, concat_features = 1)
        self.discriminator_clf = models.get_clf_head({'clf_head_ratio': self.hparams['clf_head_ratio']}, self.discriminator_emb.emb_dim + 1, 1)

        self.disc_optimizer = torch.optim.Adam( 
            list(self.discriminator_emb.parameters()) + list(self.discriminator_clf.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )  

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, device):
        all_x = cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        classifs = self.classifier(all_z)

        clf_losses = F.cross_entropy(classifs, all_y.squeeze().long(), reduction = 'none')

        disc_input = {
            'img': all_x['img'],
            'concat': all_y.unsqueeze(-1)
        }
        disc_out = torch.sigmoid(self.discriminator_clf(self.discriminator_emb(disc_input)))
        disc_weights = 1 + len(disc_out) * (disc_out / disc_out.sum())

        clf_loss_unweighted = clf_losses.mean()
        clf_loss_weighted = (clf_losses * disc_weights.squeeze()).mean()
        
        self.optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        
        if (self.update_count.item() % 2): # update adversary     
            (-clf_loss_weighted).backward()       
            self.disc_optimizer.step()
        else: # update classifier
            clf_loss_weighted.backward()
            self.optimizer.step()

        self.update_count += 1
        
        return {'update_count': self.update_count.item(), 'loss': clf_loss_weighted.item(), 'loss_unweighted': clf_loss_unweighted.item(), 
                'max_weight': disc_weights.max().item(), 'min_weight': disc_weights.min().item()}

class FairALM(ERM):
    '''
    FairALM: Augmented Lagrangian Method for Training Fair Models with Little Regret
    https://arxiv.org/pdf/2004.01355.pdf
    '''
    def __init__(self, num_classes, hparams):
        super(FairALM, self).__init__(num_classes, hparams)
        self.hparams = hparams
        self.threshold = self.hparams['fairalm_threshold']        
        self.surrogate_fns = [
            lambda y,t: tpr_surrogate(y,t, threshold = self.threshold, surrogate_fn = self.hparams['fairalm_surrogate']),
            lambda y,t: fpr_surrogate(y,t, threshold = self.threshold, surrogate_fn = self.hparams['fairalm_surrogate'])
        ]
        self.register_buffer('lag_mult', torch.zeros(len(self.surrogate_fns) * len(Constants.group_vals[hparams['protected_attr']])).float())
        self.lag_mult.requires_grad = False
        self.eta = self.hparams['fairalm_eta']

    def update(self, minibatches, device):    
        if str(self.lag_mult.device) != device:
            self.lag_mult = self.lag_mult.to(device)
        
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        all_classifs = torch.cat(classifs)
        targets = [yi for _, yi in minibatches]
        all_targets = torch.cat(targets)
        all_losses = []
        lag_increments = []
        penalty = 0.

        for i in range(nmb):
            all_losses.append(F.cross_entropy(classifs[i], targets[i], reduction = 'none'))
            if targets[i].sum() == 0 or targets[i].sum() == len(targets[i]):
                for c, fn in enumerate(self.surrogate_fns):
                    lag_increments.append(0)
                continue
            for c, fn in enumerate(self.surrogate_fns):
                lag_ind = i * len(self.surrogate_fns) + c
                grp_val = fn(classifs[i], targets[i])
                all_val = fn(all_classifs, all_targets)
                penalty += (self.lag_mult[lag_ind] + self.eta) * grp_val - (self.lag_mult[lag_ind] - self.eta) * all_val # conditional with marginal
                lag_increments.append((self.eta * (grp_val - all_val)).item())

        clf_loss = torch.mean(torch.cat(all_losses))
        final_loss = clf_loss + penalty

        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()

        self.lag_mult = self.lag_mult + (torch.tensor(lag_increments)).to(device).float()

        return {'loss': final_loss.item(), 'clf_loss': clf_loss.item(), 'max_mult': self.lag_mult.max().item(), 'min_mult': self.lag_mult.min().item()}

class JTT(ERM):
    '''
    Just Train Twice
    https://arxiv.org/pdf/2107.09044.pdf
    '''
    def __init__(self, num_classes, hparams):
        super(JTT, self).__init__(num_classes, hparams)
        self.erm_model_folder = Path(hparams['JTT_ERM_model_folder'])
        self.erm_model_args = json.load((self.erm_model_folder/'args.json').open('r'))
        assert self.erm_model_args['task'] == hparams['task']
        assert self.erm_model_args['algorithm'] == 'ERM'
        assert self.erm_model_args['data_type'] == 'normal'
        assert self.erm_model_args['val_fold'] == hparams['val_fold']
        assert self.erm_model_args['dataset'] == hparams['dataset']
        self.erm_model = ERM(num_classes, self.erm_model_args) # keep this on CPU
        self.erm_model.load_state_dict(torch.load(self.erm_model_folder/'model.pkl'))
        self.thres = hparams['JTT_threshold']
        self.weight = hparams['JTT_weight']

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        with torch.no_grad():
            pred_y = torch.nn.Softmax(dim = 1)(self.erm_model.predict(all_x))[:, 1] >= self.thres
        upweight_mask = pred_y != all_y
        weights = torch.ones(upweight_mask.size()).to(device)
        weights[upweight_mask] = self.weight
        
        loss = (weights * F.cross_entropy(self.predict(all_x), all_y.squeeze().long(), reduction = 'none')).sum()/len(weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'n_misclass': upweight_mask.sum().item()}