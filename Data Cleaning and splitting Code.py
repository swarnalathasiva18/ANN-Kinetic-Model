# Data Loading from drive

from google.colab import drive
drive.mount('/content/drive')
pip install optuna
pip install torchmetrics

#importing required lib
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import r2_score

##taking the dataset upto 3 seconds and checking the data types
df.drop(df.index[510000:], inplace = True)
df.shape
(df.select_dtypes(exclude = 'float64')).columns

# removing str type
rows_with_strings = df[pd.to_numeric(df['LVG-C6H10O5'], errors='coerce').isna()]
df = df.drop(rows_with_strings.index)

#converting the dataset to floating numbers
df['LVG-C6H10O5']=df['LVG-C6H10O5'].astype(float)
df['src_LVG-C6H1']=df['src_LVG-C6H1'].astype(float)

#visualising the components' mole fraction against its rate before removing a data
for i in range (2,46):


  plt.xlabel(df.columns[i])
  plt.ylabel(df.columns[i+44])
  print(i, sep = ',')
  print(i+44)
  plt.scatter(x = df[df.columns[i]],y = df[df.columns[i+44]], s= 10)
  plt.show()

# from the graphs it is clearly noticable that, 5 components don't contribute which are:
# 1)CH2OH 2) C2H5  3) CH2CO 4) CH2CHO 5)AlCH2-C7H7

df.drop(df.columns[[11,55,14,58,18,62,19,63,30,74]], axis=1, inplace=True)

# Removing data with mole fraction of N2 = 1

df_new = df[(df['N2'] != 1)]
df.shape

# SHUFFLING THE DATA

df = df_new.sample(frac=1).reset_index(drop=True)
df.shape

#PLOTTING MOLE FRACTION VERSUS TIME

for i in range (2,41):

  plt.ylabel(df.columns[i])
  plt.xlabel('Time')
  plt.scatter(x = df['#time'],y = df[df.columns[i]], s= 1)
  plt.show()

#PLOTTING RATE VERSUS TIME

for i in range (41,81):

  plt.ylabel(df.columns[i])
  plt.xlabel('Time')
  plt.scatter(x = df['#time'],y = df[df.columns[i]], s= 10)
  plt.show()

# visualization of data, components mole fraction against its rate after reomving the data

for i in range (2,41):

  plt.xlabel(df.columns[i])
  plt.ylabel(df.columns[i+39])
  plt.scatter(x = df[df.columns[i]],y = df[df.columns[i+39]], s= 10)
  plt.show()

#splitting the whole data into features and targets

df_x = df.iloc[:, 2:41]
df_y = df.iloc[:, 41:]

# scaling techniques

for i in df-x.columns:
  df_x[i] = (df_x[i]-df_x[i].min())/(df_x[i].max()-df_x[i].min())

#removing components with negative mole fractions

Clmns = df_x.columns
for i in Clmns:
  ind = df[df_x[i]<0].index
  df.drop(ind, inplace = True)
df.shape

#dividing the dataset into 3 parts training , validation and testing

df_x_train = df_x.iloc[0:30000, :]
df_y_train = df_y.iloc[0:30000, :]

df_x_val = df_x.iloc[30000:45000, :]
df_y_val = df_y.iloc[30000:45000, :]

df_x_test = df_x.iloc[45000:,]
df_y_test = df_y.iloc[45000:,]

#transforming into np arrays and into tensors

#TRAINING DATASET
x_np = df_x_train.to_numpy()
y_np = df_y_train.to_numpy()
x_train = torch.from_numpy(x_np)
y_train = torch.from_numpy(y_np)

#VALIDATING DATASET
x_np = df_x_val.to_numpy()
y_np = df_y_val.to_numpy()
x_val = torch.from_numpy(x_np)
y_val = torch.from_numpy(y_np)

#TEST DATASET
x_np = df_x_test.to_numpy()
y_np = df_y_test.to_numpy()
x_test = torch.from_numpy(x_np)
y_test = torch.from_numpy(y_np)




                    # Transforming and loading the dataset into customized NN

train_data = torch.utils.data.TensorDataset(x_train, y_train)
dataloader_train = DataLoader(train_data, batch_size = 750)

val_data = torch.utils.data.TensorDataset(x_val, y_val)
dataloader_val = DataLoader(val_data, batch_size = 1000)

test_data = torch.utils.data.TensorDataset(x_test, y_test)
dataloader_test = DataLoader(test_data, batch_size = df_x_test.shape[0])

df_y_val.tail(100)
