import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
#os.chdir("../../..")

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#from cProfile import label
#import numpy as np
#import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.metrics import mean_squared_error

import chart_studio.plotly as py
import chart_studio

#import torch
#import torch.nn as nn

import time, math
from datetime import date

#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
#from pytorch_forecasting import Baseline, BaseModel, SMAPE
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


#python -W ignore dl_2_Temporal_Fusion_Transformer.py

#split_data --------------------------------------------------------------------------------------
def split_data(stock, frac_test, lookback, size_train):
#def split_data(stock, frac_test, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
        #print('data_raw[',index,':', index + lookback, ']:', data_raw[index: index + lookback])
    
    data = np.array(data)
    #test_set_size = int(np.round(frac_test*data.shape[0]))
    #train_set_size = data.shape[0] - (test_set_size)
    
    train_set_size=size_train

    x_train = data[:train_set_size, :-1, :] #not including the last element
    y_train = data[:train_set_size, -1, :] #only last element

    #print('\ntrain_set_size:', train_set_size)
    #print('x_train.shape:', x_train.shape)
    #print('x_train.ndim:', x_train.ndim)
    #print('y_train.shape:', y_train.shape)
    #print('y_train.ndim:', y_train.ndim)
    #print('x_train:',x_train)
    #print('y_train:',y_train)

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

#Metrics RMSE and MAPE -------------------------------------
def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))                   
    return rmse

def calculate_mape(y_true, y_pred): 
    """
    Mean Absolute Percentage Error (MAPE)
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)    
    mape = np.mean(np.abs((y_true-y_pred) / y_true))*100    
    return mape

#GPU confugurations ----------------------------------------------------------------------------
#Check if using GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n---GPU settings -----------------------------------------')
print('index of the currently selected device:',device)
print('torch.cuda.is_available():',torch.cuda.is_available()) #Check CUDA package and version
print('torch.cuda.device_count():',torch.cuda.device_count()) #Get the number of GPUs available
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0),'') #Get name of the device
print('---------------------------------------------------------')

#Read data --------------------------------------------------------------
#filepath = './stock_data_from_yahoo_finance/IBM.csv'
filepath = './stock_data/IBM_2006-01-01_to_2018-01-01.xls'
data = pd.read_csv(filepath)
data = data.sort_values('Date')
data['Date'] = pd.to_datetime(data['Date'])

print(data.head(2))
print('data.shape:',data.shape)

# Add a time_idx (an sequence of consecutive integers that goes from min to max date) ---
# add time index
#data["time_idx"] = data["Date"].dt.year * 12 + data["Date"].dt.month
#data["time_idx"] -= data["time_idx"].min()

# add a time index that is incremented by one for each time step
data = (data.merge((data[['Date']].drop_duplicates(ignore_index=True)
.rename_axis('time_idx')).reset_index(), on = ['Date']))
print(data.head(10))
print('data.shape:',data.shape)

# Add additional features --------------------------------------------------------------------------------------------
data["month"] = data.Date.dt.month.astype(str).astype("category")  # categories have be strings
data["day_of_week"] = data.Date.dt.dayofweek.astype(str).astype("category")  # categories have be strings
data["week_of_year"] = data.Date.dt.isocalendar().week.astype(str).astype("category")  # categories have be strings
#data["log_num_sold"] = np.log(data.num_sold + 1e-8)
#data["avg_volume_by_country"] = data.groupby(["time_idx", "country"], observed=True).num_sold.transform("mean")
#data["avg_volume_by_store"] = data.groupby(["time_idx", "store"], observed=True).num_sold.transform("mean")
#data["avg_volume_by_product"] = data.groupby(["time_idx", "product"], observed=True).num_sold.transform("mean")
print(data.head(10))
print('data.shape:',data.shape)
#print(data.tail(10))



#Calculate MACD -------------------------------------------------------------------
#50 days moving ava=erage
#data_ma = data['Close'].ewm(span=50, adjust=False, min_periods=50).mean()
#print('data_ma:', data_ma.head(2))

#apply moving average (50 points) 
data_close=data['Close'].to_frame()
window_size=50
data_close['Close_SMA50'] = data['Close'].rolling(window=window_size, center=False).mean() # calculating simple moving average using .rolling(window).mean()
#first 50 points are ignore because they are used for averaging
#Do NOT remove NULL/NA points since we need all data points for comparison.
#data_close.dropna(inplace=True) # removing all the NULL values using dropna() method

print('data:', data.head(4))
print('\ndata_close:', data_close.head(4))
print('data_close.shape:',data_close.shape)

#k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
k=data['Close'].to_frame()
d=data['Close'].to_frame()
k['Ewm_12'] = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
d['Ewm_26'] = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

#plot price vs time ---------------------------------------------------------------
sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data[['Close']],label='Original')
plt.plot(data_close[['Close_SMA50']], label='Moving Average (50 days)')
plt.plot(k[['Ewm_12']], label='EMA (12 days) [K-line]')
plt.plot(d[['Ewm_26']], label='EMA (26 days) [D-line]')
#plt.plot(data_ma[['Close']], color='red',label='MA')
plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
plt.title("IBM Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.legend()
#plt.show()
plt.savefig('tft_1_price_vs_time.png')

#Prepare train/test set -----------------------------------------------------------------------
frac_test=0.1 #percentage of the data for testing/validation
size_test = test_set_size = int(np.round(frac_test*data.shape[0]))
size_train = data.shape[0] - (size_test)

train = data.iloc[:size_train]
test = data.iloc[size_train:]

print('train::',train.tail(5))
print('test::',test.head(5))
print('len(train):',len(train))
print('len(test):',len(test))

#Create dataloaders --------------------------------------------------------------------------------------------
#Convert dataframe into PyTorch Forecasting TimeSeriesDataSet
max_prediction_length = 302 #days 
max_encoder_length = train.Date.nunique()
training_cutoff = train["time_idx"].max() - max_prediction_length #we will validate on 2020
print('max_prediction_length=',max_prediction_length)
print('max_encoder_length=',max_encoder_length)
print('training_cutoff=',training_cutoff)
#print('len=',len(data[lambda x: x.time_idx <= training_cutoff].drop('time_idx', axis = 1)))
#print('train::',train.tail(5))
#print('len=',len(train[lambda x: x.time_idx <= training_cutoff].drop('time_idx', axis = 1)))
#print('train::',train.tail(5))

training = TimeSeriesDataSet(
    train[lambda x: x.time_idx <= training_cutoff],
    #train,
    #data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    #group_ids=["country", "store", "product"], 
    group_ids=["Name"], 
    min_encoder_length=max_prediction_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    #static_categoricals=["country", "store", "product"],
    static_categoricals=["Name"],
    time_varying_known_categoricals=["month", "week_of_year", "day_of_week"], 
                                     #"is_holiday",
                                     #"is_holiday_lead_1", "is_holiday_lead_2",
                                     #"is_holiday_lag_1", "is_holiday_lag_2"],
    #variable_groups={"is_holiday": ["is_holiday"]},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        #"num_sold", "log_num_sold", "avg_volume_by_country",
        #"avg_volume_by_store", "avg_volume_by_product"
        "Open", "High", "Low", "Close", "Volume"
    ],
    target_normalizer=GroupNormalizer(
        #groups=["country", "store", "product"], transformation="softplus"
        groups=["Name"], transformation="softplus"
    ),  # use softplus and normalize by group
    #categorical_encoders={
        #'week_of_year':NaNLabelEncoder(add_nan=True)
    #},
    #lags={'num_sold': [7, 30, 365]},
    #lags={'Close': [7, 30, 365]},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series ----
validation = TimeSeriesDataSet.from_dataset(training, train, predict=True, stop_randomization=True)

# create dataloaders for model ---------------------------------------------------------------------
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

#Create baseline model ------------------------------------------------------------------------------------------
#Calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
base_resol=(actuals - baseline_predictions).abs().mean().item()
print('base_resol:',base_resol)

#Train the Temporal Fusion Transformer model with Pytorch Lightning ---------------------------------------------
#Find optimal learning rate
# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)

# Configure network and trainer --------------------------------------------------------------------------------------------
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    #max_epochs=1,
    max_epochs=30,
    gpus=0,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# find optimal learning rate ---------------------------------
#res = trainer.tuner.lr_find(
#    tft,
#    train_dataloaders=train_dataloader,
#    val_dataloaders=val_dataloader,
#    max_lr=10.0,
#    min_lr=1e-6,
#)
#print(f"suggested learning rate: {res.suggestion()}")
##fig = res.plot(show=True, suggest=True)
#plt.savefig('tft_2_suggested_learning_rate.png')


'''
# fit network ------------------------------
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

#save model ----------------------------------------
torch.save(tft.state_dict(), './models/tft.pth')
'''

'''
#Hyperparameter tuning -----------------------------------------------------------------------------------------------
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=True,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("./models/tft_hy_tunning.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)
'''


































'''
#Create dataloaders for model -----------------------------------------------------------------------------------------------------
#A naive model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)

#actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
#baseline_predictions = Baseline().predict(val_dataloader)
#(actuals - baseline_predictions).abs().mean().item()

#sm = SMAPE()
#print(f"Median loss for naive prediction on validation: {sm.loss(actuals, baseline_predictions).mean(axis = 1).median().item()}")


#Training and Evaluation
PATIENCE = 30
MAX_EPOCHS = 120
LEARNING_RATE = 0.03
OPTUNA = False
#OPTUNA = True

early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-2, patience=PATIENCE, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    devices=1, 
    accelerator="gpu",
    ###gpus=1,
    max_epochs=MAX_EPOCHS,
    enable_model_summary=True,
    gradient_clip_val=0.25,
    limit_train_batches=10,  # coment in for training, running valiation every 30 batches
    fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    lstm_layers=2,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.2,
    hidden_continuous_size=8,
    output_size=1,  # 7 quantiles by default
    loss=SMAPE(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4
)

tft.to(device)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
'''

'''
if OPTUNA:
    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=50,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )
'''    

'''
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader, mode="prediction")
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

sm = SMAPE()
print(f"Validation median SMAPE loss: {sm.loss(actuals, predictions).mean(axis = 1).median().item()}")
'''



'''
x_train, y_train, x_test, y_test = split_data(price, frac_test, lookback, size_train)
#x_train, y_train, x_test, y_test = split_data(price, frac_test, lookback)

x_train_sma50, y_train_sma50, x_test_sma50, y_test_sma50 = split_data(price_sma50, frac_test, lookback, size_train)
x_train_ewm12, y_train_ewm12, x_test_ewm12, y_test_ewm12 = split_data(price_ewm12, frac_test, lookback, size_train)
x_train_ewm26, y_train_ewm26, x_test_ewm26, y_test_ewm26 = split_data(price_ewm26, frac_test, lookback, size_train)

print('size_test=',size_test)
print('size_train',size_train,'\n')
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

print('\nx_train_sma50.shape = ', x_train_sma50.shape)
print('y_train_sma50.shape = ', y_train_sma50.shape)
print('x_test_sma50.shape = ', x_test_sma50.shape)
print('y_test_sma50.shape = ', y_test_sma50.shape)
'''