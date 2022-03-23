import torch
import os
import numpy as np
from PIL import Image
from cxr_fairness.data import Constants, process
import pandas as pd
from torchvision import transforms
import pickle
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

def load_df(env, val_fold, only_frontal = False, query_str = None):
    assert(isinstance(val_fold, str))
    df = pd.read_csv(Constants.df_paths[env])
    func = process.get_process_func(env)
    df = func(df, only_frontal)
    if query_str is not None:
        df = df.query(query_str)
    train_folds = [i for i in df.fold_id.unique() if i not in ['test', val_fold]]
    ans = {
        'train': df[df.fold_id.isin(train_folds)].reset_index(drop = True),
        'val': df[df.fold_id == val_fold].reset_index(drop = True),
        'test': df[df.fold_id == 'test'].reset_index(drop = True)
    }
    assert(len(ans[i]) > 0 for i in ans)
    return ans

def get_dataset(dfs_all, env, split = None, concat_group = False, protected_attr = None, 
            imagenet_norm = True, augment = 0, use_cache = False, subset_label = None, smaller_label_set = False):
    if split in ['val', 'test']:
        assert(augment in [0, -1])
    
    if augment == 1: # image augmentations
        image_transforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomRotation(10),     
                            transforms.RandomResizedCrop(size = 224, scale = (0.75, 1.0)),
                        transforms.ToTensor()]
    elif augment == 0: 
        image_transforms = [transforms.ToTensor()]
    elif augment == -1: # only resize, just return a dataset with PIL images; don't ToTensor()
        image_transforms = []        
   
    if imagenet_norm and augment != -1:
        image_transforms.append(transforms.Normalize(Constants.IMAGENET_MEAN, Constants.IMAGENET_STD))             
    
    datasets = []       
    if split is not None:    
        splits = [split]
    else:
        splits = ['train', 'val', 'test']
        
    dfs = [dfs_all[i] for i in splits]        
        
    for c, s in enumerate(splits):
        cache_dir = Path(Constants.cache_dir)/ f'{env}/'
        cache_dir.mkdir(parents=True, exist_ok=True)
        datasets.append(ConcatWrapper(
                AllDatasetsShared(dfs[c], label_set = Constants.take_labels if smaller_label_set else Constants.take_labels_all,
                                transform = transforms.Compose(image_transforms), split = split, 
                                cache = use_cache, cache_dir = cache_dir, subset_label = subset_label),
                  concat_group = concat_group, protected_attr = protected_attr) 
                            )
                
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
        ds.dataframe = pd.concat([i.dataframe for i in datasets])
    
    return ds

class ConcatWrapper(Dataset):
    def __init__(self, ds, concat_group = False, protected_attr = None):
        super().__init__()
        self.ds = ds
        self.concat_group = concat_group
        self.protected_attr = protected_attr

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, meta = self.ds[idx]
        if self.concat_group:
            if self.protected_attr == 'sex':
                concat = torch.tensor([meta['sex'] == 'M']).float()
            else:
                concat = torch.zeros(len(Constants.group_vals[self.protected_attr])).float()
                concat[Constants.group_vals[self.protected_attr].index(meta[self.protected_attr])] = 1.
        else:
            concat = torch.tensor([]).float()
        x_new = {'img': x, 
                 'concat': concat}
        return x_new, y, meta    

class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, label_set = Constants.take_labels, transform=None, split = None, cache = True, 
                cache_dir = '', subset_label = None):
        super().__init__()
        self.dataframe = dataframe
        self.label_set = label_set
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.split = split
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self.subset_label = subset_label # (str) select one label instead of returning all Constants.take_labels

    def get_cache_path(self, cache_dir, meta):
        path = Path(meta['path'])
        if meta['env'] in ['PAD', 'NIH']:
            return cache_dir / (path.stem + '.pkl')
        elif meta['env'] in ['MIMIC', 'CXP']:
            return (cache_dir / '_'.join(path.parts[-3:])).with_suffix('.pkl')  
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        cache_path = self.get_cache_path(self.cache_dir, item)
        
        if self.cache and cache_path.is_file():
            img, _, _ = pickle.load(cache_path.open('rb'))
            meta = item.to_dict()
        else:            
            img = np.array(Image.open(item["path"]))

            if img.dtype == 'int32':
                img = np.uint8(img/(2**16)*255)
            elif img.dtype == 'bool':
                img = np.uint8(img)
            else: #uint8
                pass

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)            
            elif len(img.shape)>2:
                img = img[:,:,0]
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2) 

            img = Image.fromarray(img)
            resize_transform = transforms.Resize(size = [224, 224])            
            img = transforms.Compose([resize_transform])(img)            
            meta = item.to_dict()
            
            if self.cache:
                pickle.dump((img, [], meta), cache_path.open('wb')) # empty list for compatability
        
        if self.subset_label:
            label = int(self.dataframe[self.subset_label].iloc[idx])
        else:
            label = torch.FloatTensor(np.zeros(len(self.label_set), dtype=float))
            for i in range(0, len(self.label_set)):
                if (self.dataframe[self.label_set[i].strip()].iloc[idx].astype('float') > 0):
                    label[i] = self.dataframe[self.label_set[i].strip()].iloc[idx].astype('float')
        
        if self.transform is not None: # apply image augmentations after caching
            img = self.transform(img)        
                
        return img, label, meta
            

    def __len__(self):
        return self.dataset_size
