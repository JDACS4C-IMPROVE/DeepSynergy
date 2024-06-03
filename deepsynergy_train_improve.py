""" Train model for synergy prediction.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# Model-specific imports
import os
import pickle
import gzip
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout

# [Req] Imports from preprocess script
from deepsynergy_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
app_train_params = []
model_train_params = []
train_params = app_train_params + model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"] 
# or
# metrics_list = ["mse", "acc", "recall", "precision", "f1", "auc", "aupr"]

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# [Req]
def run(params):
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]

    train_data_path = params["ml_data_outdir"] + "/" + train_data_fname
    val_data_path = params["ml_data_outdir"] + "/" + val_data_fname
    
    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    layers = [8182,4096,1] 
    epochs = 1000 
    act_func = tf.nn.relu 
    dropout = 0.5 
    input_dropout = 0.2
    eta = 0.00001 
    norm = 'tanh' 
    #data_file = 'data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)
    data_file = 'data_test_fold0_tanh.p'
    with open(data_file, 'rb') as f:
        file = pickle.load(f)

    #file = gzip.open(data_file, 'rb')
    X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
    file.close()
    
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
    set_session(tf.Session(config=config))

    model = Sequential()
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(input_dropout)))
        elif i==len(layers)-1:
            model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
        else:
            model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
            model.add(Dropout(float(dropout)))
        model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))

    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------

    hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
    val_loss = hist.history['val_loss']
    model.reset_states()

    average_over = 15
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
    epo = np.argmin(smooth_val_loss)

    hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
    test_loss = hist.history['val_loss']
    # -----------------------------
    # Save model
    # -----------------------------

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    val_pred = model.predict(y_test)
    val_true = y_test
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores




# [Req]
def main(args):
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="deepsynergy_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])