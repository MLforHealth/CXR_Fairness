# Downloading the Data
## MIMIC-CXR
1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We recommend downloading from the GCP bucket:

```
gcloud auth login
mkdir MIMIC-CXR-JPG
gsutil -m rsync -d -r gs://mimic-cxr-jpg-2.0.0.physionet.org MIMIC-CXR-JPG
```

2. In order to obtain gender information for each patient, you will need to obtain access to [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/). Download `core/patients.csv.gz` and `core/admissions.csv.gz` and place the files in the `MIMIC-CXR-JPG` directory.

## CheXpert
1. Sign up with your email address [here](https://stanfordmlgroup.github.io/competitions/chexpert/).

2. Download either the original or the downsampled dataset (we recommend the downsampled version - `CheXpert-v1.0-small.zip`) and extract it.

3. Register for an account and download the CheXpert demographics data [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf). 


# Data Processing
1. In `cxr_fairness/data/Constants.py`, update `image_paths` to point to the two directories that you downloaded, and `CXP_details` to be the path to the CheXpert demographics file.

2. Run `python -m cxr_fairness.data.preprocess.preprocess`. 

3. (Optional) If you are training a lot of models, it _might_ be faster to cache all images to binary 224x224 files on disk. This is especially true if you are using non-downsized versions of the datasets. In this case, you should update the `cache_dir` path in `cxr_fairness/data/Constants.py` and then run `python -m cxr_fairness.data.preprocess.cache_data`, optionally parallelizing over `--env_id {0, 1}` for speed. To use the cached files, pass `--use_cache` to `train.py`.