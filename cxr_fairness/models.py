import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import timm
import cxr_fairness.data.Constants as Constants

class CBR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.block = [
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() 
        ]
        if max_pool:
            self.block.append(nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.block = nn.Sequential(*self.block)
        
    def forward(self, x):
        return self.block(x)   

class CBR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            CBR_Block(3, 32, 7),
            CBR_Block(32, 64, 7),
            CBR_Block(64, 128, 7),
            CBR_Block(128, 256, 7),
            CBR_Block(256, 512, 7, False)            
            # nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.emb_dim = 512
        
    def forward(self, x):
        return self.model(x)   

class EmbModel(nn.Module):
    def __init__(self, emb_type, pretrain, concat_features = 0):
        super().__init__()
        self.emb_type = emb_type
        self.pretrain = pretrain
        self.concat_features = concat_features
        
        if emb_type == 'densenet':
            model = models.densenet121(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.children())[:-1]) #https://discuss.pytorch.org/t/densenet-transfer-learning/7776/2
            self.emb_dim = model.classifier.in_features
        elif emb_type == 'resnet':
            model = models.resnet50(pretrained=pretrain)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.emb_dim = list(model.children())[-1].in_features
        elif emb_type == 'vision_transformer':
            self.encoder  = timm.create_model('vit_deit_small_patch16_224', pretrained= pretrain, num_classes=0)
            self.emb_dim = self.encoder.num_features
        elif emb_type == 'CBR':
            self.encoder = CBR() 
            self.emb_dim = self.encoder.emb_dim

        self.n_outputs = self.emb_dim + concat_features      
        
    def forward(self, inp):
        if isinstance(inp, dict): # dict with image and additional feature(s) to concat to embedding
            x = inp['img']
            concat = inp['concat']
            assert(concat.shape[-1] == self.concat_features)
        else: # tensor image
            assert(self.concat_features == 0)
            x = inp

        x = self.encoder(x).squeeze(-1).squeeze(-1)
        if self.emb_type == 'densenet':
            x = F.relu(x)
            x = F.avg_pool2d(x, kernel_size = 7).view(x.size(0), -1)
            
        if isinstance(inp, dict):
            x = torch.cat([x, concat], dim = -1)
        
        return x 
   
def get_featurizer(hparams):
    if hparams['concat_group']:
        if hparams['protected_attr'] == 'sex':
            n_concat_features = 1
        else:
            n_concat_features = len(Constants.group_vals[hparams['protected_attr']])
    else:
        n_concat_features = 0
    return EmbModel(hparams['model'], pretrain = True, concat_features = n_concat_features)

def get_clf_head(hparams, featurizer_n_outputs, n_classes):
    n_hidden = int(featurizer_n_outputs//hparams['clf_head_ratio'])
    return nn.Sequential(
        nn.Linear(featurizer_n_outputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_classes)
    )
    