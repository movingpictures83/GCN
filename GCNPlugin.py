#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

current_directory = os.getcwd()        
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input, layers
from math import sqrt
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

from spektral.layers import GCNConv

from baselines.gcn import gcn


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class GCNPlugin:
 def input(self, inputfile):
  self.inputfile = inputfile
 def run(self):
     pass
 def output(self, outputfile):
  # ### Parameters
  n_hours = 72
  K = 24 
  masked_value = 1e-10
  split_1 = 0.8
  split_2 = 0.9
  sigma2 = 0.1
  epsilon = 0.5

  # ===== model parameters ======
  gcn_unit1 = 32
  gcn_unit2 = 16
  lstm_units = 32
  dropout = 0
  learning_rate = 5e-4
  decay_steps = 10000
  decay_rate = 0.95
  PATIENCE = 500
  EPOCHS = 5
  BATCH = 512
  from preprocess.BaselinePrerocess import gcn_process
  from preprocess.graph import graph_topology
  from tensorflow.keras.models import load_model
  from sklearn.metrics import mean_squared_error as mse
  from sklearn.metrics import mean_absolute_error as mae
  from math import sqrt

  train_X_mask, val_X_mask, test_X_mask, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler = gcn_process(n_hours, K, masked_value, split_1, split_2, self.inputfile)
  print(train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, train_ws_y.shape, val_ws_y.shape, test_ws_y.shape)
  train_adj_mat, val_adj_mat, test_adj_mat = graph_topology(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y), self.inputfile)
  print(train_adj_mat.shape, val_adj_mat.shape, test_adj_mat.shape)
  n_nodes = train_adj_mat.shape[-1]
  n_timesteps = train_X_mask.shape[-1]
  # n_nodes, n_timesteps, gcn1, gcn2, lstm_unit
  model, GCNConv = gcn(gcn1=gcn_unit1,
                     gcn2=gcn_unit2,
                     n_nodes=n_nodes,
                     n_timesteps=n_timesteps,
                     lstm_unit=lstm_units,
                     dropout=dropout,
                     masked_value=masked_value
                    )
  model.summary()
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate)
  model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=['mae']
             )
  es = EarlyStopping(monitor='val_mae', mode='min', verbose=2, patience=PATIENCE)
  #mc = ModelCheckpoint('saved_models/WaLeF_gcn.h5', 
  mc = ModelCheckpoint(outputfile, 
                     monitor='val_mae', 
                     mode='min', 
                     verbose=2, 
                     custom_objects={'GCNConv': GCNConv},
                     save_best_only=True
                    )


  model.fit([train_X_mask, train_adj_mat], train_ws_y,
          validation_data=([test_X_mask, test_adj_mat], test_ws_y),
          epochs=EPOCHS,
          batch_size=BATCH,
          verbose=2,
          shuffle=True,
          callbacks=[es, mc]
         )
  #saved_model = load_model('saved_models/WaLeF_gcn.h5', custom_objects={'GCNConv': GCNConv})
  saved_model = load_model(outputfile, custom_objects={'GCNConv': GCNConv})
  yhat = saved_model.predict([test_X_mask, test_adj_mat])
  inv_yhat = ws_scaler.inverse_transform(yhat)
  inv_y = ws_scaler.inverse_transform(test_ws_y)
  print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)
  print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
  print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
  wss = ['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
  inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
  inv_y_reshape = inv_y.reshape((-1, 24, 4))
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    plt.plot(error[:, -1, i], linewidth=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Time steps', fontsize=18)
    plt.ylabel('Actual Error (ft)', fontsize=18)
    plt.title('{}'.format(wss[i]), fontsize=18)
    plt.text(9000, -0.6, 'MAE = {}'.format(float("{:.3f}".format(sum(abs(error[:, -1, i]))/len(error)))), fontsize=12)
    plt.show()
  up_thre = 0.5
  low_thre = -0.5
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] > up_thre) + np.sum(error[:, -1, i] < low_thre))
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] > up_thre))
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] < low_thre))


  # In[ ]:
  from preprocess.BaselinePrerocess import baseline_process, gcn_process, baseline_process_for_gate_predictor
  from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process
  from preprocess.GraphTransformerPrerocess import graph_global_transformer_local_process
  from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process_for_gate_predictor
  from preprocess.graph import graph_topology, graph_topology_5
  from tensorflow.keras.models import load_model
  from postprocess.threshold import flood_threshold, drought_threshold, flood_threshold_t1, drought_threshold_t1
  from postprocess.errors import estimate_error
  from sklearn.metrics import mean_squared_error as mse
  from sklearn.metrics import mean_absolute_error as mae
  from spektral.layers import GCNConv
  from baselines.tcn import TCN
  from preprocess.helper import series_to_supervised
  import time
  # ====== preprocessing parameters ======
  n_hours = 72
  K = 24
  masked_value = 1e-10
  split_1 = 0.8
  split_2 = 0.9
  sigma2 = 0.1
  epsilon = 0.5
  train_X_mask_gcn, val_X_mask_gcn, test_X_mask_gcn, train_ws_y_gcn, val_ws_y_gcn, test_ws_y_gcn, scaler, ws_scaler = gcn_process(n_hours, K, masked_value, split_1, split_2, self.inputfile)
  print(train_X_mask_gcn.shape, val_X_mask_gcn.shape, test_X_mask_gcn.shape, train_ws_y_gcn.shape, val_ws_y_gcn.shape, test_ws_y_gcn.shape)
  train_adj_mat, val_adj_mat, test_adj_mat = graph_topology(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y), self.inputfile)
  print(train_adj_mat.shape, val_adj_mat.shape, test_adj_mat.shape)
  #saved_models = ['WaLeF_gcn']
  #for i in range(len(saved_models)):
  #print("===================== {} =====================".format(saved_models[i]))

  # load model and make prediction
  #model = load_model('saved_models/{}.h5'.format(saved_models[i]), custom_objects={'GCNConv': GCNConv})
  model = load_model(outputfile, custom_objects={'GCNConv': GCNConv})
  start_time = time.perf_counter()
  yhat = model.predict([test_X_mask_gcn, test_adj_mat])
  end_time = time.perf_counter()
  used_time = end_time - start_time
  print(f"Usded time: {used_time} seconds")

  # inverse transformation
  inv_yhat = ws_scaler.inverse_transform(yhat)
  inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
  inv_y = ws_scaler.inverse_transform(test_ws_y)
  inv_y = inv_y[:, [0, 24, 48, 72]]

  # compute MAE and RMSE
  print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
  print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))

  errors = inv_yhat - inv_y
  print('Numbers of over/under estimate:', estimate_error(errors))


