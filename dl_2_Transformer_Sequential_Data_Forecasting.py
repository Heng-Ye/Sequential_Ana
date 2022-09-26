import torch
from torch import nn
import numpy as np
import pandas as pd

import os, datetime
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score

plt.style.use('seaborn')

#Split data --------------------------------------------------
def split_data(stock, frac_test, lookback, size_train):
#def split_data(stock, frac_test, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    #test_set_size = int(np.round(frac_test*data.shape[0]))
    #train_set_size = data.shape[0] - (test_set_size)
    
    train_set_size=size_train

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

#Normalization ------------------------------------------------------------------------------------------------------------
def normalize(df):
    result = df.copy()

    result[['Open', 'High', 'Low', 'Close', 'Volume']] = result[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change()
    result.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

    min_return = min(result[['Open', 'High', 'Low', 'Close']].min(axis=0))
    max_return = max(result[['Open', 'High', 'Low', 'Close']].max(axis=0))

    result['Open'] = (result['Open'] - min_return) / (max_return - min_return)
    result['High'] = (result['High'] - min_return) / (max_return - min_return)
    result['Low'] = (result['Low'] - min_return) / (max_return - min_return)
    result['Close'] = (result['Close'] - min_return) / (max_return - min_return)

    min_vol = result['Volume'].min(axis=0)
    max_vol = result['Volume'].max(axis=0)
    result['Volume'] = (result['Volume'] - min_vol) / (max_vol - min_vol)

    return result




#GPU confugurations ----------------------------------------------------------------------------
#Check if using GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n---GPU settings -----------------------------------------')
print('index of the currently selected device:',device)
print('torch.cuda.is_available():',torch.cuda.is_available()) #Check CUDA package and version
print('torch.cuda.device_count():',torch.cuda.device_count()) #Get the number of GPUs available
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0),'') #Get name of the device
print('---------------------------------------------------------')

#Read data ----------------------------------------------------------------------------
features=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
filepath = './stock_data_from_yahoo_finance/IBM.csv'
#data = pd.read_csv(filepath)
data = pd.read_csv(filepath, delimiter=',', usecols=features)

#Replace 0 with the previous non-zero value to avoid dividing by 0 for later analysis
data['Volume'].replace(to_replace=0, method='ffill', inplace=True) 
data = data.sort_values('Date')
print('data.head:',data.head(10))
print('data.shape:',data.shape)
#---------------------------------------------------------------------------------------

#Data smoothing --------------------------------------------------------------------------------------------------------
# Apply moving average with a window of 10 days to all columns (the first 9 elements are gone for averaging)
data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].rolling(10).mean() 

# Drop all rows with NaN values
data.dropna(how='any', axis=0, inplace=True) 
print(data.head(10))
print('data.shape:',data.shape)
#-----------------------------------------------------------------------------------------------------------------------

#Data Prep --------------------------------------------------------------------------
print('data:', data.head(2))
data=normalize(data)
print('norm data:', data.head(2))
#------------------------------------------------------------------------------------

# Split the data sequentially into 80% train, 10% valid and 10% test ----------------
#Create training, validation and test split

# Sort on date and find the rows -10% and -20% from the end
times = sorted(data.index.values)
print(data.index.values)
last_10pct = sorted(data.index.values)[-int(0.1*len(times))] # Last 10% of series
last_20pct = sorted(data.index.values)[-int(0.2*len(times))] # Last 20% of series

# Split train, valid and test
df_train = data[(data.index < last_20pct)]  # Training data are 80% of total data
df_val = data[(data.index >= last_20pct) & (data.index < last_10pct)]
df_test = data[(data.index >= last_10pct)]

# Remove date column
print(df_train.head(5))
df_train.drop(columns=['Date'], axis=1, inplace=True)
df_val.drop(columns=['Date'], axis=1, inplace=True)
df_test.drop(columns=['Date'], axis=1, inplace=True)
print(df_train.head(5))
#axis=0 for rows and 1 for columns.

# Convert pandas columns into arrays
train_data = df_train.values
val_data = df_val.values
test_data = df_test.values
print('Training data shape: {}'.format(train_data.shape))
print('Validation data shape: {}'.format(val_data.shape))
print('Test data shape: {}'.format(test_data.shape))

#Plot the data of daily percent deltas -------------------------------------------------------------------------------
fig = plt.figure(figsize=(15,12))
st = fig.suptitle("Data Preparation", fontsize=20)
st.set_y(0.95)

ax1 = fig.add_subplot(211)
ax1.plot(np.arange(train_data.shape[0]), df_train['Close'], label='Training data')

ax1.plot(np.arange(train_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]), df_val['Close'], label='Validation data')

ax1.plot(np.arange(train_data.shape[0]+val_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['Close'], label='Test data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Closing Returns')
ax1.set_title("Close Price", fontsize=18)
ax1.legend(loc="best", fontsize=12)


ax2 = fig.add_subplot(212)
ax2.plot(np.arange(train_data.shape[0]), df_train['Volume'], label='Training data')

ax2.plot(np.arange(train_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]), df_val['Volume'], label='Validation data')

ax2.plot(np.arange(train_data.shape[0]+val_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), df_test['Volume'], label='Test data')
ax2.set_xlabel('Date')
ax2.set_ylabel('Normalized Volume Changes')
ax2.set_title("Volume", fontsize=18)
ax2.legend(loc="best", fontsize=12)

plt.savefig('1_Norm_price_vs_time.png')