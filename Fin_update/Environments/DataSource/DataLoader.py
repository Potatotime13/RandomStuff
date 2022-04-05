import numpy as np
import random as rn
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

def create_dataset(batches,stocks,days,depth,noise=0.5):
    '''
    creates random sinus timeseries per stock and adds noise
    the last value of the series is used to calc the label: (up/down)
    '''
    x_vals = np.zeros([batches,stocks,days,depth])
    y_vals = np.zeros([batches,stocks])
    for i in range(stocks):
        data = np.expand_dims(np.sin(np.pi*(rn.random()+np.linspace(0,2,num=days+1))), axis=1)
        for j in range(batches):
            data_tmp = data + noise * (np.random.rand(days+1,depth)*2-1)
            x_vals[j,i,:,:] = data_tmp[:days,:].copy()
            if data_tmp[days,3]/data_tmp[days-1,3] >= 1:
                y_vals[j,i] = 1
            else:
                y_vals[j,i] = 0
    return x_vals, y_vals

def check_data(data,label,num_plots=9):
    '''
    plots timeseries and labels, red: down ; blue: up
    '''
    width = int(num_plots**0.5)
    length = int(num_plots/width)+ num_plots % width    
    fig = plt.figure(figsize=(width*3,length*2))
    for i in range(num_plots):
        index_b = rn.randint(1,data.shape[0]-1)
        index_s = rn.randint(0,data.shape[1]-2)
        if label[index_b, index_s] > 0.8:
            plt.subplot(length, width, i+1)
            plt.plot(data[index_b, index_s,:,3])
        else:
            plt.subplot(length, width, i+1)
            plt.plot(data[index_b, index_s,:,3],color='red')
    plt.show()

def stock_loader_yahoo(stocks,market,start,end):
    '''
    downloading stock data using yfinance and converting it to the format
    [stock,time,feature] where feature is ['Open','High','Low','Close','Adj Close']
    drops stocks if they have nans in the trading timewindow
    '''
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    perm = ['Open','High','Low','Close','Adj Close']
    data = yf.download(stocks,'2019-01-01','2022-01-01')
    data.dropna(axis=1, inplace=True)
    st_list = data.columns.get_level_values(1).unique().to_list()
    stocks = intersection(stocks,st_list)
    data_out = yf.download(market,start=start,end=end).loc[:,perm]
    len_dat = len(stocks)+1

    for st, j in zip(stocks, range(len(stocks))):
        tmp = yf.download(st,start=start,end=end).loc[:,perm]
        tmp = tmp.add_prefix(st)
        data_out = pd.merge(data_out, tmp, how='left',left_index=True,right_index=True)
    # numpy reshape
    data_out.fillna(1, inplace=True)
    numpy_out = np.zeros((len_dat,data_out.shape[0],data_out.shape[1]//len_dat))
    for i in range(len_dat):
        numpy_out[i,:,:] = data_out.iloc[:,i*5:(i+1)*5].to_numpy()
    return numpy_out