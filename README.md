# Improving the Fairness of Chest X-ray Classifiers

Benchmarking the performance of group-fair and minimax-fair methods on two chest x-ray datasets, with an auxiliary investigation into label bias in MIMIC-CXR. For more details please see our [CHIL 2022 paper](https://arxiv.org/abs/2203.12609).

## Contents
  - [Setting Up](#setting-up)
    - [1. Environment and Prerequisites](#1-environment-and-prerequisites)
    - [2. Obtaining and Preprocessing the Data](#2-obtaining-and-preprocessing-the-data)
  - [Main Experimental Grid](#main-experimental-grid)
    - [1. Running Experiments](#1-running-experiments)
    - [2. Model Selection and Bootstrapping](#2-model-selection-and-bootstrapping)
    - [3. Aggregating Results](#3-aggregating-results)
  - [Auxiliary Experiments](#auxiliary-experiments)
    - [Radiologist Labelled Dataset](#radiologist-labelled-dataset)
    - [Proxy Labels](#proxy-labels)
  - [Citation](#citation)

## Setting Up
### 1. Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:MLforHealth/CXR_Fairness.git
cd CXR_Fairness/
conda env create -f environment.yml
conda activate cxr_fairness
```

### 2. Obtaining and Preprocessing the Data
See [DataSources.md](DataSources.md) for detailed instructions.

## Main Experimental Grid
### 1. Running Experiments
To reproduce the experiments in the paper which involve training grids of models using different debiasing methods, use `cxr_fairness/sweep.py` as follows:

```
python -m cxr_fairness.sweep launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `cxr_fairness/experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `cxr_fairness/launchers.py` (i.e. `slurm` or `local`).

Sample bash scripts showing the command can also be found in `bash_scripts/`.

The `ERM` experiment should be ran first. The remaining experiments can be ran in any order, except `JTT` should be ran after `ERM` and after updating the path in its experiment, and `Bootstrap` should not be ran until the next step. 

Alternatively, a single model can also be trained at once by calling `cxr_fairness/train.py` with the appropriate arguments, for example:

```
python -m cxr_fairness.train \
    --algorithm DistMatch \
    --distmatch_penalty_weight 5.0 \
    --match_type mean \
    --batch_size 32 \
    --data_type balanced \
    --dataset CXP \
    --output_dir {output_dir} \
    --protected_attr sex \
    --task "No Finding" \
    --val_fold 0 
```

### 2. Model Selection and Bootstrapping
After all experiments have finished, run the `notebooks/get_best_model_configs.ipynb` notebook to select the best models across hyperparameter settings.

Then, run the `Bootstrap` experiment using `cxr_fairness.sweep` as shown above, updating the path in `experiments.py` appropriately. 

### 3. Aggregating Results
We provide the following notebooks in the `notebooks` folder to create figures shown in the paper:
- `agg_results_single_target.ipynb`: Creates two main result figures (performance metrics and comparison with Balanced ERM) for a task and dataset.
- `adv_perf_graph.ipynb`: Creates the figure showing performance of group fairness methods as a function of the loss term weighting.


## Auxiliary Experiments
### Radiologist Labelled Dataset
We provide the radiologist labelled dataset at `aux_data/MIMIC_CXR_Rad_Labels.xlsx`, and we analyze the dataset using `notebooks/rad_labels.ipynb`.

### Proxy Labels
We link the x-rays in MIMIC-CXR with hospital stay information in MIMIC-IV by querying MIMIC-IV through Google BigQuery in a [Colab Notebook](https://colab.research.google.com/drive/1MgOmE2NPhkKD5e2fhLJ0ES7tHpUtJgwF?usp=sharing). The resulting data can then be downloaded, and is merged with the model predictions and analyzed using `notebooks/proxy_label_graphs.ipynb`.


## Citation
If you use this code in your research, please cite the following publication:
```
@article{zhang2022improving,
  title={Improving the Fairness of Chest X-ray Classifiers},
  author = {Zhang, Haoran and Dullerud, Natalie and Roth, Karsten and Oakden-Rayner, Lauren and Pfohl, Stephen Robert and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:2203.12609},
  year={2022}
}
```
