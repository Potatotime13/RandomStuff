import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Models.CustomLayers.CLayers import *

class DTML(keras.Model):
    '''
    lstm / attention hybrid model for stock price prediction
    TO DEFINE:
    -number of stocks
    -hidden representation of a timeseries
    -number of attention heads -> head width is than hidden_representation/num_heads
    -drop out rate of the mlp after the attention
    -number of observed timesteps
    '''
    def __init__(self, num_stocks, num_dense, num_heads, rate, window):
        super().__init__()
        self.lstm_multi = LSTM_Multipath(num_dense,num_stocks)
        self.lstm_norm = LSTM_Normalization(num_dense)
        self.multi_con = Multi_Context(0.1)
        self.multi_head = layers.MultiHeadAttention(num_heads=num_heads, key_dim=int(num_dense/num_heads), attention_axes=(0, 1))
        self.context_tr = Context_Transformer(rate, num_dense)
        self.predictor = layers.Dense(1,activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.01),bias_regularizer=keras.regularizers.l2(0.01))
        self.window = window

    def call(self, inputs):
        x = self.lstm_multi(inputs)
        x = self.lstm_norm(x)
        x = self.multi_con(x)
        x_a = self.multi_head(x,x)
        x = self.context_tr(x, x_a)
        x = tf.squeeze(self.predictor(x), [2])
        return x

    def summary(self):
        super().summary(expand_nested=True)

    def decide(self, predictions, money, prices, port):
        '''
        simple rule buy every stock with >0.7 and sell <0.3

        -INPUTS:
        predictions: model predictions of increasing price probability per stock
        money: current liquidity in the trading simulation
        prices: selling prices of the current day
        port: current portfolio positions 

        -OUTPUTS:
        buy: binary vector with ones for buy stock at this index
        sell: binary vector with ones for sell stock at this index
        '''
        buy = predictions > 0.5
        sell = predictions < 0.5
        sell = sell*port # sell all selling-stocks in the portfolio
        money += np.sum(sell*prices)
        per_stock = money / max(np.sum(buy),1)
        buy = buy*(per_stock/prices).astype(int) # use nearly all money to buy buying-stocks
        return buy, sell

    def generate_data(self, timeseries, mid):
        '''
        timeseries 3D matrix with [stock,time,feature]
        feature format [open, high, low, close, adjclose]
        num_stocks number of stocks with market reference included
        output = array[batch,stock,time,feature], array[batch,stock,time,feature], ...
        '''
        # helper functions
        def mov_avg(dm,ts,ks):
            return np.sum(dm[:,ts-ks:ts,4],axis=1)/(ks*dm[:,ts-1,4])-1
        
        # data arrays
        data_ts = timeseries.copy()
        num_stocks = timeseries.shape[0]
        data_tmp = np.zeros((timeseries.shape[0],timeseries.shape[1],11))
        movs = [5,10,15,20,25,30]

        # transform stock data discribed in table 3 in paper
        data_tmp[:,:,:3] = data_ts[:,:,:3]/data_ts[:,:,3:4]-1
        data_tmp[:,1:,3:5] = data_ts[:,1:,3:5]/data_ts[:,:-1,3:5]-1
        for t in range(30,timeseries.shape[1]):
            for id, mov in enumerate(movs):
                data_tmp[:,t-1,5+id] = mov_avg(data_ts,t,mov)

        # create labels and prices
        data_labels = (data_tmp[:,:,3].copy() > 0)
        prices_full = data_ts[:,:,3].copy()

        # cut labels and prices
        data_labels = np.delete(data_labels,slice(30),axis=1)
        data_labels = np.delete(data_labels,0,axis=0)
        prices_full = np.delete(prices_full,slice(29),axis=1)
        prices_full = np.delete(prices_full,0,axis=0)
        prices_full = np.delete(prices_full,-1,axis=1)

        # midpoint adjustment
        mid -= 29

        # cut trading prices
        prices_trade = prices_full[:,mid:]

        # create and cut datasets
        # cut dataset to match labels and mov avgs
        data_tmp = np.delete(data_tmp,slice(29),axis=1)
        data_tmp = np.delete(data_tmp,-1,axis=1)

        # training set TODO fill forward ???
        data_train = data_tmp[:,:mid,:]
        rand_ind = np.random.permutation(np.arange(self.window-1,mid))
        train_in = np.zeros((rand_ind.shape[0],num_stocks,self.window,data_train.shape[2]))
        train_out = np.zeros((rand_ind.shape[0],data_labels.shape[0]))
        for i, rn in enumerate(rand_ind):
            train_in[i,:,:,:] = data_train[:,rn+1 -self.window: rn+1,:]
            train_out[i,:] = data_labels[:,rn]
        
        # trading set TODO fill forward
        data_trade = data_tmp[:,mid-self.window+1:,:]
        test_in = np.zeros((data_trade.shape[1]+1-self.window,num_stocks,self.window,data_trade.shape[2]))
        test_out = np.zeros((data_trade.shape[1]+1-self.window,data_labels.shape[0]))
        for j in range(self.window-1, test_in.shape[0]):
            test_in[j+1 -self.window,:,:,:] = data_trade[:,j+1 -self.window: j+1,:]
            test_out[j+1 -self.window,:] = data_labels[:,j]

        return train_in, train_out, test_in, test_out, prices_trade, prices_full   