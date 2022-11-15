#!/usr/bin/env python
# coding: utf-8

# In[186]:


#!pip install pmdarima


# In[185]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.filters.hp_filter import hpfilter
from tqdm import tqdm_notebook
from itertools import product
from pmdarima.arima.utils import ndiffs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


# In[137]:


import yfinance as yf

ticker = "SPY"
stock_data = yf.download(ticker, start='2000-01-02', end="2022-10-26")


# In[138]:


stock_data.head()


# In[139]:


stock_data.info()


# In[140]:


stock_data.isnull().sum()


# In[141]:


#sns.lineplot(x = 'Date', y = 'Open',data = stock_data)


# In[142]:


#sns.lineplot(x = 'Date', y = 'Close',data = stock_data)


# In[143]:


#sns.lineplot(x = 'Date', y = 'Adj Close',data = stock_data)


# In[144]:


#sns.lineplot(x = 'Date', y = 'Volume',data = stock_data)


# In[145]:


#sns.lineplot(x = 'Date', y = 'High',data = stock_data)


# In[146]:


#sns.lineplot(x = 'Date', y = 'Low',data = stock_data)


# Trend and seasonality present in the data. For better understanding, let's do an STL decomposition
# Decomposition like seasonal decomposition outputs separating seasonality, trend, level, resid

# # STL Decomposition

# In[147]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[148]:


start_date = datetime(2000,1,2)
end_date = datetime(2022,10,26)
stock = stock_data[start_date:end_date]


# In[149]:


stock = stock.reset_index()
stock["Date"] = pd.to_datetime(stock["Date"])
stock.set_index('Date', inplace=True)


# In[150]:


for (columnName, columnData) in stock_data.iteritems():
    stocks = stock.loc[:, stock.columns == columnName]
    result = seasonal_decompose(stocks, model='multiplicative', period=5)
    fig = result.plot()


# Hodrick-Prescott Filter (HPF)

# In[151]:


stockdata = pd.DataFrame(stock_data.Close)
Close_cycle, Close_trend = hpfilter(stockdata, lamb=129600)
stockdata['cycle'] = Close_cycle
stockdata['trend'] = Close_trend

stockdata.plot(figsize=(10, 5), title='Close Pollutant Plot of Cycle and Trend')


# Smoothing the data so that underlying trends in the data can be easily determined and also to detect significant changes in direction

# # SMOOTHING

# Moving Average

# In[152]:


def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    
#Smooth by the previous quarter (90 days)
plot_moving_average(stock_data.Close, 90)

#Smooth by the previous 6 months (30 days)
plot_moving_average(stock_data.Close, 180)

#Smooth by previous year (365 days)
plot_moving_average(stock_data.Close, 365, plot_intervals=True)


# Exponential smoothing

# In[153]:


def exponential_smoothing(series, alpha):

    result = [series[0]] 
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
  
def plot_exponential_smoothing(series, alphas):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);

plot_exponential_smoothing(stock_data.Close, [0.05, 0.3])


# Double exponential smoothing

# In[154]:


def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result

def plot_double_exponential_smoothing(series, alphas, betas):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series.values, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)
    
plot_double_exponential_smoothing(stock_data.Close, alphas=[0.9, 0.02], betas=[0.9, 0.02])


# autocorrelation : whether or not pairs of data show autocorrelation(error terms in a time series are correlated between periods)

# adfuller test : test statistic < Critical Value and p-value < 0.05 then null hypothesis can be rejected. Time series does not have a unit root and is stationary.

# # Stationarity check

# In[155]:


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dfoutput['p-value'] < 0.05:
        print("Stationary")
    else:
        print("Non-Stationary")


# In[156]:


adf_test(stock_data['Open']) #Non-stationary based on output
adf_test(stock_data['Close']) #Non-stationary based on output
adf_test(stock_data['Adj Close']) #Non-stationary based on output
adf_test(stock_data['High']) #Non-stationary based on output
adf_test(stock_data['Low']) #Non-stationary based on output
adf_test(stock_data['Volume']) #stationary based on output


# Converting non-stationary data to stationary by using square root transformation.
# Towards stationarity : Box-cox transformation or Differencing

# # Using square root transformation

# In[157]:


df_log=np.sqrt(stock_data['Open'])
df_diff=df_log.diff().dropna()


# In[158]:


df_log=np.sqrt(stock_data['Close'])
df_diff1=df_log.diff().dropna()


# In[159]:


df_log=np.sqrt(stock_data['Adj Close'])
df_diff2=df_log.diff().dropna()


# In[160]:


df_log=np.sqrt(stock_data['High'])
df_diff3=df_log.diff().dropna()


# In[161]:


df_log=np.sqrt(stock_data['Low'])
df_diff4=df_log.diff().dropna()


# Checking stationarity again after the transformation.

# In[162]:


result=adfuller(df_diff)
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))


# Comparing the before version of timeseries with transformed version of it

# In[165]:


plt.figure(figsize=(15,8))
plt.plot(df_diff1,label="tranformed")
plt.plot(stock_data['Close'],label="original")
plt.tick_params(
    axis='x',        
    which='both',   
    bottom=False,      
    top=False,        
    labelbottom=False)
plt.legend()
plt.show() 


# Train-Test Split

# In[121]:


from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits = 3)


# In[122]:


stock_data.sort_index(inplace=True)
X = stock_data.drop(labels=['Close'], axis=1)
y = stock_data['Close']


# In[123]:


for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[124]:


y_train.groupby('Date').mean().plot()
y_test.groupby('Date').mean().plot()


# # AUTO CORRELATION

# P,Q can be calculated by PACF,ACF graphs for classic ARIMA.
# 
# partial autocorrelation graph of the data(PACF), cutoff value of PACF is the P value
# Autocorrelation graph(ACF), cutoff value of ACF is the Q value

# # Modelling - AUTO ARIMA

# d-value is 1 after making 'Close' stationary

# In[169]:


d_val = ndiffs(stock_data['Close'], test='adf')
print('Arima D-value:', d_val)


# In[170]:


#splitting the data to train and test sets based on Ntest value
#last 30 days
Ntest = 30
train = stock_data.iloc[:-Ntest]
test = stock_data.iloc[-Ntest:]
train_idx = stock_data.index <= train.index[-1]
test_idx = stock_data.index > train.index[-1]

#Define auto-arima to find best model
model = pm.auto_arima(train['Close'],
                      d = d_val,
                      start_p = 0,
                      max_p = 15,
                      start_q = 0,
                      max_q = 15,
                      stepwise=False,
                      max_order=30,
                      trace=True)


# In[171]:


model.get_params()


# In[182]:


def plot_result(model, data, col_name, Ntest):
    
    params = model.get_params()
    d = params['order'][1]
    
    #In sample data prediction
    train_pred = model.predict_in_sample(start=d, end=-1)
    #out of sample prediction
    test_pred, conf = model.predict(n_periods=Ntest, return_conf_int=True)
    
    #plotting real values, fitted values and prediction values
    fig, ax= plt.subplots(figsize=(15,8))
    ax.plot(data[col_name].index, data[col_name], label='Actual Values')
    ax.plot(train.index[d:], train_pred, color='green', label='Fitted Values')
    ax.plot(test.index, test_pred, label='Forecast Values')
    ax.fill_between(test.index, conf[:,0], conf[:,1], color='red', alpha=0.3)
    ax.legend()
    
    #evaluating the model using RMSE and MAE metrics
    y_true = test[col_name].values
    rmse = np.sqrt(mean_squared_error(y_true,test_pred))
    mae = mean_absolute_error(y_true,test_pred)
    mape = mean_absolute_percentage_error(y_true,test_pred)

    return rmse, mae, mape


# In[184]:


rmse , mae, mape = plot_result(model, stock_data, 'Close', Ntest=30)
print('Root Mean Squared Error: ', rmse)
print('Mean Absolute Error: ', mae)
print('Mean Absolute Percentage Error: ', mape)


# In[181]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




