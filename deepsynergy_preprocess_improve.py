""" Preprocess data to generate datasets for the prediction model.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports
import pickle 
import gzip

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
app_preproc_params = []
model_preproc_params = [
    {"name": "datasets",
     "type": str,
     "default": "ALMANAC",
     "help": "datasets to use",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------
def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt) 

# [Req]
def run(params: Dict):
    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    params = frm.build_paths(params)  

    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # Load X data (feature representations)
    # ------------------------------------------------------
    norm = 'tanh'
    test_fold = 0
    val_fold = 1
    file = gzip.open('X.p.gz', 'rb')
    X = pickle.load(file)
    file.close()
    # ------------------------------------------------------
    # Load Y data 
    # ------------------------------------------------------
    #contains synergy values and fold split (numbers 0-4)
    labels = pd.read_csv('labels.csv', index_col=0) 
    #labels are duplicated for the two different ways of ordering in the data
    labels = pd.concat([labels, labels]) 

    #indices of training data for hyperparameter selection: fold 2, 3, 4
    idx_tr = np.where(np.logical_and(labels['fold']!=test_fold, labels['fold']!=val_fold))
    #indices of validation data for hyperparameter selection: fold 1
    idx_val = np.where(labels['fold']==val_fold)

    #indices of training data for model testing: fold 1, 2, 3, 4
    idx_train = np.where(labels['fold']!=test_fold)
    #indices of test data for model testing: fold 0
    idx_test = np.where(labels['fold']==test_fold)

    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_train = X[idx_train]
    X_test = X[idx_test]

    y_tr = labels.iloc[idx_tr]['synergy'].values
    y_val = labels.iloc[idx_val]['synergy'].values
    y_train = labels.iloc[idx_train]['synergy'].values
    y_test = labels.iloc[idx_test]['synergy'].values


    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    if norm == "tanh_norm":
        X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
        X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
                                                            feat_filt=feat_filt, norm=norm)
    else:
        X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
        X_test, mean, std, feat_filt = normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm)

    pickle.dump((X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test), 
        open('data_test_fold%d_%s.p'%(test_fold, norm), 'wb'))

    # ------------------------------------------------------
    # [Req] Create data names for ML data
    # ------------------------------------------------------
    #train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    #val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]
    #test_data_fname = frm.build_ml_data_name(params, stage="test")  # [Req]

    #train_data_path = params["ml_data_outdir"] + "/" + train_data_fname
    #val_data_path = params["ml_data_outdir"] + "/" + val_data_fname
    #test_data_path = params["ml_data_outdir"] + "/" + test_data_fname

    # ------------------------------------------------------
    # Save ML data
    # ------------------------------------------------------
    #with open(train_data_path, 'wb+') as f:
    #    pickle.dump(train_data, f, protocol=4)

    #with open(val_data_path, 'wb+') as f:
    #    pickle.dump(val_data, f, protocol=4)
    
    #with open(test_data_path, 'wb+') as f:
    #    pickle.dump(test_data, f, protocol=4)
   

    return params["ml_data_outdir"]


# [Req]
def main(args):
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="deepsynergy_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])