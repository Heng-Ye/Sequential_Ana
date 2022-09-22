from cProfile import label
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score

import chart_studio.plotly as py
import chart_studio

import torch
import torch.nn as nn

import time, math
from sklearn.metrics import mean_squared_error
import sys

#split_data --------------------------------------------------------------------------------------
def split_data(stock, frac_test, lookback, size_train):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    train_set_size=size_train

    x_train = data[:train_set_size, :-1, :] #not including the last element
    y_train = data[:train_set_size, -1, :] #only last element
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

#LSTM Model -------------------------------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) #Tensor using CPU only 
        out, (hn, cn) = self.lstm(x.cuda(), (h0.detach().cuda(), c0.detach().cuda())) #Use CUDA GPU tensor
        out = self.fc(out[:, -1, :]) 
        return out

#GPU confugurations ----------------------------------------------------------------------------
#Check if using GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Read data ------------------------------------------------------------------------------------------------
filepath = './stock_data_from_yahoo_finance/IBM.csv'
data = pd.read_csv(filepath)
data = data.sort_values('Date')

#read argument ---------------------
lookback = int(sys.argv[1])
print('input lookback=', lookback)

#Calculate MACD -------------------------------------------------------------------
#50 days moving ava=erage
#data_ma = data['Close'].ewm(span=50, adjust=False, min_periods=50).mean()

#apply moving average (50 points) 
data_close=data['Close'].to_frame()
#window_size=50
#data_close['Close_SMA50'] = data['Close'].rolling(window=window_size, center=False).mean() # calculating simple moving average using .rolling(window).mean()
#first 50 points are ignore because they are used for averaging
#Do NOT remove NULL/NA points since we need all data points for comparison.
#data_close.dropna(inplace=True) # removing all the NULL values using dropna() method

#Data Normalization ---------------------------------------
#[1]Before scaling -----
#get pric'close' column
price = data[['Close']]
data_price=data.loc[:, 'Close']

#[2]After scaling ----------------------------------------------------------
#Scaling makes NN alg. converge faster
#change range of values without changing the shape of distribution
#range set to -1 to 1  
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

#[3]Prepare data for training --------------------------------------------------------------------------------
frac_test=0.3 #percentage of the data for testing/validation
#lookback = 450 # choose sequence length (size of sliding window)
#HY: lookback set to high number, better performance
#for example, use current data at index j as truth, data points before j (j-20 to j-1) for training the model 

#split data into train -----------------------------------------------------------------
#setup size of train and test/valid set
size_test = test_set_size = int(np.round(frac_test*data.shape[0]))
size_train = data.shape[0] - (size_test)
x_train, y_train, x_test, y_test = split_data(price, frac_test, lookback, size_train)

#[4]Prepare train and test set to pytorch tensor -----------------------
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

#Utilize GPUs for computation (CUDA tensor types) -----------
x_train=x_train.cuda()
x_test=x_test.cuda()
y_train_lstm=y_train_lstm.cuda()
y_test_lstm=y_test_lstm.cuda()

x_train=x_train.to(device)
x_test=x_test.to(device)
y_train_lstm=y_train_lstm.to(device)
y_test_lstm=y_test_lstm.to(device)

#[5]LSTM model -----------------------------------------------------------------------------------------------
input_dim = 1
hidden_dim = 32
num_layers = 2 
output_dim = 1
#num_epochs = 150
num_epochs = 500
model_lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

#Model save to GPU -----------------------------------------------------
model_lstm.cuda()
model_lstm.to(device) #build model on gpu

#Sansity check to see if model parameters are in GPU
#for m in model_lstm.parameters():
    #print(m.device) #return cuda:0

criterion = torch.nn.MSELoss(reduction='mean')
optimiser_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)

#[6]Timer setup -------------------------------------------------------
hist_lstm = np.zeros(num_epochs) #histogram to save loss for iteration
start_time = time.time()
lstm = []

#[7]Training process --------------------------------------
for t in range(num_epochs):
    #y_train_pred = model_lstm(x_train)
    tmp_y_train_pred_lstm = model_lstm(x_train)

    loss_lstm = criterion(tmp_y_train_pred_lstm, y_train_lstm)
    #print("Epoch ", t, "MSE: ", loss_lstm.item())
    hist_lstm[t] = loss_lstm.item()

    optimiser_lstm.zero_grad()
    loss_lstm.backward()
    optimiser_lstm.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
#-----------------------------------------------------------

# Model predictions ------------------------------------------------------------------------------------------
#predicted scaled price value [test set]
y_scaled_test_pred_lstm = model_lstm(x_test)
y_test_pred_lstm = pd.DataFrame(scaler.inverse_transform(y_scaled_test_pred_lstm.cpu().detach().numpy()))

y_scaled_train_pred_lstm = model_lstm(x_train)
y_train_pred_lstm = pd.DataFrame(scaler.inverse_transform(y_scaled_train_pred_lstm.cpu().detach().numpy()))

orig_lstm = pd.DataFrame(scaler.inverse_transform(y_train_lstm.cpu().detach().numpy()))
test_lstm = pd.DataFrame(scaler.inverse_transform(y_test_lstm.cpu().detach().numpy()))

# Evaluate performance ----------------------------------------------------
mape_lstm = calculate_mape(test_lstm, y_test_pred_lstm)
print('MAPE (LSTM):', mape_lstm)

# Save results --------------------------------------------------------
file_name='./lstm_lookback_optwindow/lookback_'+sys.argv[1]+'.csv'
print('Will save file in:',file_name)
file = open(file_name, 'w')
file.write(str(mape_lstm))
file.close()

