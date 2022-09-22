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

#GRU Model -------------------------------------------------------------------------------------------------
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach().cuda()))
        out = self.fc(out[:, -1, :]) 
        return out




#GPU confugurations ----------------------------------------------------------------------------
#Check if using GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\n---GPU settings -----------------------------------------')
print('index of the currently selected device:',device)
print('torch.cuda.is_available():',torch.cuda.is_available()) #Check CUDA package and version
print('torch.cuda.device_count():',torch.cuda.device_count()) #Get the number of GPUs available
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0),'') #Get name of the device
print('---------------------------------------------------------')

#Read data ------------------------------------------------------------------------------------------------
#filepath = './stock_data/AMZN_2006-01-01_to_2018-01-01.xls'
#filepath = './stock_data/IBM_2006-01-01_to_2018-01-01.xls'
filepath = './stock_data_from_yahoo_finance/IBM.csv'
#data=yf.download('AMZN',start='2020-09-01', interval='1h',  end='2022-08-29',progress=False)[['Close']]
#data=yf.download('AMZN', period='5y', interval='1d',  end='2022-08-29',progress=False)[['Close']]
data = pd.read_csv(filepath)
data = data.sort_values('Date')
print(data.head(2))
print('data.shape:',data.shape)

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

'''
# Get the 12-day EMA of the closing price
k = data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()

# Get the 26-day EMA of the closing price
d = data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

# Subtract the 26-day EMA from the 12-Day EMA to get the MACD
macd = k - d

# Get the 9-Day EMA of the MACD for the Trigger line
macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

# Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
macd_h = macd - macd_s

# Add all of our new values for the MACD to the dataframe
data['macd'] = pd.index.map(macd)
data['macd_h'] = pd.index.map(macd_h)
data['macd_s'] = pd.index.map(macd_s)
# View our data
pd.set_option("display.max_columns", None)
#print(df)
'''

#plot price vs time ---------------------------------------------------------------
sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data[['Close']],label='Original')
plt.plot(data_close[['Close_SMA50']], label='Moving Average (50 days)')
plt.plot(k[['Ewm_12']], label='EMA (12 days) [K-line]')
plt.plot(d[['Ewm_26']], label='EMA (26 days) [D-line]')
#plt.plot(data_ma[['Close']], color='red',label='MA')
plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
plt.title("Amazon Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.legend()
#plt.show()
plt.savefig('price_vs_time.png')

#Data Normalization ---------------------------------------
#[1]Before scaling -----
#get pric'close' column
price = data[['Close']]
price_sma50=data_close[['Close_SMA50']]
price_ewm12=k[['Ewm_12']]
price_ewm26=d[['Ewm_26']]

#price_ma = data_ma[['Close']] 
#print('price.info:', price.info(),'\n')
print('\nprice[Close]:\n', price['Close'].head(5))
print('price[Close].shape:', price['Close'].shape)
print('price[Close].ndim:', price['Close'].ndim)

data_price=data.loc[:, 'Close']
print(data_price.head(5))
print('\ndata_price.shape:', data_price.shape)
print('data_price.ndim:', data_price.ndim)


#[2]After scaling ----------------------------------------------------------
#Scaling makes NN alg. converge faster
#change range of values without changing the shape of distribution
#range set to -1 to 1  
scaler = MinMaxScaler(feature_range=(-1, 1))
#price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
#print('price.shape=',price.shape,'\n')
#data_price = scaler.fit_transform(data_price.values.reshape(-1,1))
print('price[Close].head(5):', price['Close'].head(5))
#print(price.head(5))

#[3]Prepare data for training --------------------------------------------------------------------------------
frac_test=0.3 #percentage of the data for testing/validation
#for example, frac_train=0.2 meaning first 80% for training and rest 20 % for validation
lookback = 340 # choose sequence length (size of sliding window)
#HY: lookback set to high number, better performance
#for example, use current data at index j as truth, data points before j (j-20 to j-1) for training the model 

#split data into train -----------------------------------------------------------------
#setup size of train and test/valid set
size_test = test_set_size = int(np.round(frac_test*data.shape[0]))
size_train = data.shape[0] - (size_test)

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

#[4]Prepare train and test set to pytorch tensor -----------------------
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

#Utilize GPUs for computation (CUDA tensor types) -----------
x_train=x_train.cuda()
x_test=x_test.cuda()
y_train_lstm=y_train_lstm.cuda()
y_test_lstm=y_test_lstm.cuda()
y_train_gru=y_train_gru.cuda()
y_test_gru=y_test_gru.cuda()

x_train=x_train.to(device)
x_test=x_test.to(device)
y_train_lstm=y_train_lstm.to(device)
y_test_lstm=y_test_lstm.to(device)
y_train_gru=y_train_gru.to(device)
y_test_gru=y_test_gru.to(device)
print('sansity check, x_train.is_cuda:', x_train.is_cuda)
print('sansity check, x_test.is_cuda:', x_test.is_cuda)

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
    print("Epoch ", t, "MSE: ", loss_lstm.item())
    hist_lstm[t] = loss_lstm.item()

    optimiser_lstm.zero_grad()
    loss_lstm.backward()
    optimiser_lstm.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))
#-----------------------------------------------------------

#[8]GRU model -----------------------------------------------------------------------------------------------
input_dim_gru = 1
hidden_dim_gru = 32
num_layers_gru = 2 
output_dim_gru = 1
num_epochs_gru = 500

model_gru = GRU(input_dim=input_dim_gru, hidden_dim=hidden_dim_gru, output_dim=output_dim_gru, num_layers=num_layers_gru)
model_gru.cuda()
model_gru.to(device) #build model on gpu

criterion_gru = torch.nn.MSELoss(reduction='mean')
optimiser_gru = torch.optim.Adam(model_gru.parameters(), lr=0.01)

hist_gru = np.zeros(num_epochs)
start_time_gru = time.time()
gru = []

for t in range(num_epochs):
    tmp_y_train_pred_gru = model_gru(x_train)

    loss_gru = criterion_gru(tmp_y_train_pred_gru, y_train_gru)
    print("Epoch[GRU] ", t, "MSE: ", loss_gru.item())
    hist_gru[t] = loss_gru.item()

    optimiser_gru.zero_grad()
    loss_gru.backward()
    optimiser_gru.step()

training_time_gru = time.time()-start_time_gru    
print("Training time [GRU]: {}".format(training_time_gru))

#[9] Model predictions ------------------------------------------------------------------------------------------
#LSTM
y_scaled_test_pred_lstm = model_lstm(x_test)
y_test_pred_lstm = pd.DataFrame(scaler.inverse_transform(y_scaled_test_pred_lstm.cpu().detach().numpy()))

y_scaled_train_pred_lstm = model_lstm(x_train)
y_train_pred_lstm = pd.DataFrame(scaler.inverse_transform(y_scaled_train_pred_lstm.cpu().detach().numpy()))

orig_lstm = pd.DataFrame(scaler.inverse_transform(y_train_lstm.cpu().detach().numpy()))
test_lstm = pd.DataFrame(scaler.inverse_transform(y_test_lstm.cpu().detach().numpy()))

#GRU
y_scaled_test_pred_gru = model_gru(x_test)
y_test_pred_gru = pd.DataFrame(scaler.inverse_transform(y_scaled_test_pred_gru.cpu().detach().numpy()))

y_scaled_train_pred_gru = model_gru(x_train)
y_train_pred_gru = pd.DataFrame(scaler.inverse_transform(y_scaled_train_pred_gru.cpu().detach().numpy()))

orig_gru = pd.DataFrame(scaler.inverse_transform(y_train_gru.cpu().detach().numpy()))
test_gru = pd.DataFrame(scaler.inverse_transform(y_test_gru.cpu().detach().numpy()))


# Performance plots ------------------------------------------------------------------------------------------------------------------
#LSTM
sns.set_style("darkgrid")
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
#ax = sns.lineplot(x = original_lstm.index, y = original_lstm[0], label="Data", color='royalblue')
#ax = sns.lineplot(x = predict_lstm.index, y = predict_lstm[0], label="Training Prediction (LSTM)", color='tomato')
ax = sns.lineplot(x = orig_lstm.index, y = orig_lstm[0], label="Data", color='royalblue')
ax = sns.lineplot(x = y_train_pred_lstm.index, y = y_train_pred_lstm[0], label="Training Prediction (LSTM)", color='tomato')

ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist_lstm, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)

plt.savefig('Performance_LSTM.png')

#GRU
sns.set_style("darkgrid")
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = orig_gru.index, y = orig_gru[0], label="Data", color='royalblue')
ax = sns.lineplot(x = y_train_pred_gru.index, y = y_train_pred_gru[0], label="Training Prediction (GRU)", color='tomato')

ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)

plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist_lstm, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)

plt.savefig('Performance_GRU.png')

# Evaluate performance ----------------------------------------------------
rmse_lstm = calculate_rmse(test_lstm, y_test_pred_lstm) #truth, reco
mape_lstm = calculate_mape(test_lstm, y_test_pred_lstm)
mape_gru = calculate_mape(test_gru, y_test_pred_gru)
mse_lstm = mean_squared_error(test_lstm, y_test_pred_lstm)
r2_lstm = r2_score(test_lstm, y_test_pred_lstm)
ev_lstm=explained_variance_score(test_lstm, y_test_pred_lstm)
mgd_lstm=mean_gamma_deviance(test_lstm, y_test_pred_lstm)

mape_sma50 = calculate_mape(test_lstm, y_test_sma50)
mape_ewm12 = calculate_mape(test_lstm, y_test_ewm12)
mape_ewm26 = calculate_mape(test_lstm, y_test_ewm26)

#print("Train data R2 score:", r2_score(original_ytrain, train_predict))
#print("Test data R2 score:", r2_score(original_ytest, test_predict))

print('\nRMSE (LSTM):', rmse_lstm)
#print("Train data RMSE: ", math.sqrt(mean_squared_error(original, predict_lstm)))
print('MAPE (LSTM):', mape_lstm)
print('MAPE (GRU):', mape_gru)
print('MAPE (SMA50):', mape_sma50)
print('MAPE (EWM12):', mape_ewm12)
print('MAPE (EWM26):', mape_ewm26)
print('MSE (LSTM):', mse_lstm)
print('R2 (LSTM):', r2_lstm)
print('Explained Variance Score (LSTM):', ev_lstm,'\n')
print('mape_ewm12/mape_lstm:', 1.*mape_ewm12/mape_lstm)
print('mape_sma50/mape_lstm:', 1.*mape_sma50/mape_lstm)

#plot price vs time all together ---------------------------------------------------------------
#LSTM
print('y_test_pred_lstm.shape:',y_test_pred_lstm.shape)
print('y_train_pred_lstm.shape:',y_train_pred_lstm.shape)
print('size_train:', size_train)
sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)

plt.plot(data['Date'], data[['Close']],label='Original')
plt.plot(data['Date'].loc[lookback+1:size_train+lookback], y_train_pred_lstm, label='LSTM Train',color='slateblue')
#label_sma50='Moving Average (50 days): MAPE={:.2f}'.format(mape_sma50)
plt.plot(data['Date'], data_close[['Close_SMA50']], label='SMA (50 days): MAPE={:.2f}'.format(mape_sma50))
plt.plot(data['Date'], k[['Ewm_12']], label='EMA (12 days): MAPE={:.2f}'.format(mape_ewm12))
plt.plot(data['Date'], d[['Ewm_26']], label='EMA (26 days): MAPE={:.2f}'.format(mape_ewm26), color='cyan')
plt.plot(data['Date'].loc[len(y_train_pred_lstm)+lookback:], y_test_pred_lstm, label='LSTM Test: MAPE={:.2f}'.format(mape_lstm), color='red')
#plt.plot(data['Date'].loc[len(y_train_pred_lstm)+lookback:], test_lstm, label='LSTM Test')

plt.title("IBM Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.legend()
#plt.show()
plt.savefig('price_vs_time_all.png')

#GRU 
sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)

plt.plot(data['Date'], data[['Close']],label='Original')
plt.plot(data['Date'], k[['Ewm_12']], label='EMA (12 days): MAPE={:.2f}'.format(mape_ewm12))
plt.plot(data['Date'].loc[len(y_train_pred_lstm)+lookback:], y_test_pred_lstm, label='LSTM Test: MAPE={:.2f}'.format(mape_lstm))
plt.plot(data['Date'].loc[len(y_train_pred_gru)+lookback:], y_test_pred_gru, label='GRU Test: MAPE={:.2f}'.format(mape_gru), color='red')

plt.title("IBM Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price (USD)',fontsize=18)
plt.legend()
#plt.show()
plt.savefig('price_vs_time_gru.png')



