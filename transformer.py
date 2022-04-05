import random as rn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin.stock_info as si
import re

'''
Paper: 
Accurate Multivariate Stock Movement Prediction via Data-Axis 
Transformer with Multi-Level Contexts

Idea:
mid range probability [0.7,0.3] compare to vola prediction
-> if high vola and mid range : option straddle
'''

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

def data_generator_z_features(timeseries, window=10):
    '''
    timeseries 3D matrix with [stock,time,feature]
    feature format [open, high, low, close, adjclose]
    num_stocks number of stocks with market reference included
    iteration in the training loop e[1,4]
    output = array[batch,stock,time,feature], array[batch,stock,time,feature]

    Longrun: each model includes its own data generator taking a standardized input
    '''
    def mov_avg(dm,ts,ks):
        return np.sum(dm[:,ts-ks:ts,4],axis=1)/(ks*dm[:,ts-1,4])-1
    data_ts = timeseries.copy()
    num_stocks = timeseries.shape[0]
    data_tmp = np.zeros((timeseries.shape[0],timeseries.shape[1],11))
    # transform stock data discribed in table 3 in paper
    data_tmp[:,:,:3] = data_ts[:,:,:3]/data_ts[:,:,3:4]-1
    data_tmp[:,1:,3:5] = data_ts[:,1:,3:5]/data_ts[:,:-1,3:5]-1

    for t in range(30,timeseries.shape[1]):
        data_tmp[:,t-1,5] = mov_avg(data_ts,t,5)
        data_tmp[:,t-1,6] = mov_avg(data_ts,t,10)
        data_tmp[:,t-1,7] = mov_avg(data_ts,t,15)
        data_tmp[:,t-1,8] = mov_avg(data_ts,t,20)
        data_tmp[:,t-1,9] = mov_avg(data_ts,t,25)
        data_tmp[:,t-1,10] = mov_avg(data_ts,t,30)    
    data_labels = (data_tmp[:,:,3].copy() > 0)
    data_labels = np.delete(data_labels,slice(30),1)
    data_labels = np.delete(data_labels,0,0)
    data = data_tmp[:,29:-1,:].copy()
    # portionate
    shape = data.shape
    end = shape[1]
    mid = (shape[1] // 4) * 3
    # train
    rand_ind = np.random.permutation(mid)
    trainset = np.zeros((mid,num_stocks,window,shape[2]))
    trainsol = np.zeros((mid,num_stocks-1))
    for i in range(mid):
        trainset[i,:,:,:] = data[:,rand_ind[i]:rand_ind[i]+window,:]
        trainsol[i,:] = data_labels[:,rand_ind[i]+window-1]
    # validate
    testset = np.zeros((end-mid,num_stocks,window,shape[2]))
    testsol = np.zeros((end-mid,num_stocks-1))
    for j in range(end-mid):
        testset[j,:,:,:] = data[:,mid-window+j+1:mid+j+1,:]
        testsol[j,:] = data_labels[:,mid+j]
    
    return trainset,trainsol,testset,testsol, timeseries[1:,mid+29:-1,4].copy().T


'''
#######################################################################################
################################ MODEL ################################################
#######################################################################################
'''

class LSTM_Attention(layers.Layer):
    '''
    Combining the lstm hidden states h along the time series using the formula (2)
    The operation is batch compatible, input is a 3D tensor of all [batches,model_d,time]
    for transpose operations permutations are needed as we only want to transpors model_d and time
    '''
    def __init__(self):
        super(LSTM_Attention, self).__init__()

    def call(self, inputs):
        a_i = tf.exp(tf.matmul(inputs,tf.transpose(inputs[:,0:1,:], perm=[0,2,1])))
        a_i = tf.divide(a_i,tf.reduce_sum(a_i))
        outputs = tf.matmul(tf.transpose(inputs,perm=[0,2,1]), a_i)
        return outputs

class LSTM_Multipath(layers.Layer):
    '''
    Multi LSTM models running in parallel througt the timeseries of the stocks
    the first timeseries is always the overall market
    before the time states are fed to the lstm models they will be transformed using a dense layer
    the input dim is 4D as [batches,stocks,time,state]
    the output after the transformation is [batches,stocks,time,model_d]
    the lstm works with the input [batches,time,features]
    '''
    def __init__(self, dense_output, num_paths):
        super(LSTM_Multipath, self).__init__()
        self.dense_output = dense_output
        self.num_paths = num_paths

    def build(self, inp_sh):
        self.transformer = layers.Dense(self.dense_output,activation='tanh')
        self.lstm_models = [layers.LSTM(self.dense_output, return_sequences=True, return_state=True) for _ in range(self.num_paths)]
        self.attention = LSTM_Attention()

    def call(self, inputs):
        x = self.transformer(inputs)
        x_list = []
        for i in range(x.shape[1]):
            tmp, _, _= self.lstm_models[i](x[:,i,:,:])
            tmp = self.attention(tmp)
            x_list.append(tmp)
        outputs = tf.stack(x_list, axis=1)
        return outputs

class LSTM_Normalization(layers.Layer):
    def __init__(self, lstm_dim):
        super(LSTM_Normalization, self).__init__()
        self.lstm_dim = lstm_dim

    def build(self, inp_sh):
        self.factor_gamma = self.add_weight('gammas',[1,inp_sh[1],inp_sh[2],1])
        self.factor_beta = self.add_weight('betas',[1,inp_sh[1],inp_sh[2],1])

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1])
        mean = tf.expand_dims(mean, axis=1)
        std = tf.expand_dims(tf.sqrt(var), axis=1)
        outputs = tf.math.add(
                    tf.math.multiply(self.factor_gamma,
                    tf.math.divide(tf.math.subtract(inputs,mean),std)),
                    self.factor_beta)
        return outputs

class Multi_Context(layers.Layer):
    def __init__(self,market_weight):
        super(Multi_Context, self).__init__()
        self.market_weight = market_weight

    def build(self, inp_sh):
        self.factor_beta = self.market_weight

    def call(self, inputs):
        outputs = tf.add(inputs[:,1:,:,:],tf.multiply(self.factor_beta, inputs[:,0:1,:,:]))
        return outputs

class Context_Transformer(layers.Layer):
    def __init__(self, rate, num_dense):
        super(Context_Transformer, self).__init__()
        self.rate = rate
        self.num_dense = num_dense

    def build(self, inp_sh):
        self.d1 = layers.Dense(4*self.num_dense, activation='relu')
        self.d2 = layers.Dense(self.num_dense)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerdrop = tf.keras.layers.Dropout(self.rate)

    def call(self, input1, input2):
        inp_sum = tf.squeeze(tf.math.add(input1,input2), [3])
        mlp = self.d1(inp_sum)
        mlp = self.d2(mlp)
        out = tf.math.tanh(tf.math.add(inp_sum, mlp))
        out = self.layerdrop(out)
        outputs = self.layernorm(tf.squeeze(input2, [3]) + out)
        return outputs

class DTML(keras.Model):
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
        for j in range(self.window-1, test_in.shape[1]):
            test_in[j+1 -self.window,:,:,:] = data_trade[:,j+1 -self.window: j+1,:]
            test_out[j+1 -self.window,:] = data_labels[:,j]

        return train_in, train_out, test_in, test_out, prices_trade, prices_full   

    def decide(self, predictions, money, prices, port):
        '''
        simple rule buy every stock with >0.7 and sell <0.3
        '''
        buy = predictions > 0.7
        sell = predictions < 0.3
        sell = sell*port # sell all selling-stocks in the portfolio
        money += np.sum(sell*prices)
        per_stock = money / np.sum(buy)
        buy = buy*(per_stock/prices).astype(int) # use nearly all money to buy buying-stocks
        return buy, sell


class DummyModel(keras.Model):
    def __init__(self, num_stocks, num_dense, num_heads, rate):
        super().__init__()
        self.lstm_multi = LSTM_Multipath(num_dense,num_stocks)
        self.lstm_norm = LSTM_Normalization(num_dense)
        self.predictor = layers.Dense(1,activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.01),bias_regularizer=keras.regularizers.l2(0.01))
        self.multi_con = Multi_Context(0.1)

    def call(self, inputs):
        x = self.lstm_multi(inputs)
        x = self.lstm_norm(x)
        x = self.multi_con(x)        
        x = self.predictor(x[:,:,:,0])
        return x[:,:,0]

    def summary(self):
        super().summary(expand_nested=True) 

# TODO
class TradeEnv():
    def __init__(self) -> None:
        self.stocks = []
        self.time = []
        self.costs = 0.6
        self.separator = 0.75

    def trading_simulation(self, model, tradeepochs, tradeset, tradesol, prices):
        '''
        model: Keras Model
        epochs: training epochs [train, trade]
        trainset: pre shuffled and cutted timeseriesdata [input, labels]
        tradeset: linear timeseries data, in the future of training data, with overlap of size (window)
        prices: daily prices per stock [day,stock]

        Longrun: pretraining optional, transaction cost, create simulation class(automated Data laoding and environment creation)
                    buy sell decision with additional heuristics
        '''
        def execute_trade(buy, sell, port, prices):
            port += buy - sell
            revenue = np.sum(sell * prices) - np.sum(buy * prices)
            revenue -= (np.sum(sell>0) + np.sum(buy>0)) * self.cost
            return revenue, port

        portfolio = np.zeros(prices.shape[1])
        money = 100000
        money_s = money

        # trading
        for i in range(tradeset.shape[0]):
            pred = model.predict(tradeset[i:i+1,:,:,:])[0,:]
            bu, se = model.decide(pred, money, prices[i,:], portfolio)
            rev, portfolio = execute_trade(bu, se, portfolio, prices[i,:])
            money += rev        
            if i > 0 and i % 20 == 0:
                model.fit(tradeset[i-20:i,:,:,:],tradesol[i-20:i,:], batch_size=4, epochs=tradeepochs)
        
        return (money + np.sum(prices[i,:] * portfolio))/money_s

    def load_data(self):
        with open('test.npy', 'rb') as f:
            data = np.load(f)
        mid = int(data.shape[1] * self.seperator)
        return data, mid

# TODO bug multi con for batch
# TODO training with cross entropy
if False:
    stocks = open('stocks.txt', 'r')
    stocks = stocks.read()
    stocks = re.split(' ', stocks)

    data = stock_loader_yahoo(stocks,'SPY','2010-01-01','2022-01-01')
    a,b,c,d,e = data_generator_z_features(data)
    model = DTML(4,64,4,0.15)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(a,b,epochs=20)

if False:
    with open('test.npy', 'rb') as f:
        data = np.load(f)
    #a,b,c,d,e = data_generator_z_features(data)
    #a,b = create_dataset(10000,100,10,11,noise=0.3)

    num_stocks = data.shape[0]

    model = DTML(num_stocks,64,4,0.05,10)

    train_in, train_out, test_in, test_out, prices_trade, prices_full = model.generate_data(data,2244)
    test = data[1:,2244:,3]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.FalseNegatives()])
    model.fit(a,b,batch_size=4, epochs=2)
    #result = trading_simulation(model,20,2,a,b,c,d,e)
    #print(result)
