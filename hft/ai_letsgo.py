import numpy as np
import pandas as pd
import os
import unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from keras.models import Sequential
from keras.layers import Input, CuDNNLSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
import tensorflow as tf
import keras
import keras.backend as K
from hft.handle_data import get_file_names, get_raw_book, get_midpoints, get_trades, get_dirlist, get_raw_trades


def create_trade_list(stock, days=10):
    timedelta = pd.Timedelta(microseconds=10000)
    dir_list, path = get_dirlist(book=True)
    dir_list_t, path_t = get_dirlist(book=False)
    files = get_file_names(stock, dir_list)
    files_t = get_file_names(stock, dir_list_t)
    trade_list = []
    prices = np.arange(0, 39, 2)
    for i in tqdm(range(len(files[:days]))):
        book = get_raw_book(path+files[i])
        if not book.empty:
            trades = get_trades(path_t+files_t[i])
            groups = pd.DataFrame(index=trades.index,
                                  columns=np.arange(0, len(prices)))
            for j in range(len(trades)):
                ind = book.index[book.index < trades.index[j]][-1]
                book_tmp = book.loc[ind, :].to_numpy()
                difference_array = np.absolute(
                    book_tmp[prices]-trades.iloc[j, 0])
                index = difference_array.argmin()
                groups.iloc[j, index] = trades.iloc[j, 1]
            trade_list.append(groups.fillna(0))
    return pd.concat(trade_list)


def create_trade_prob(stock, days=10):
    dir_list, path = get_dirlist(book=True)
    dir_list_t, path_t = get_dirlist(book=False)
    files = get_file_names(stock, dir_list)
    files_t = get_file_names(stock, dir_list_t) 
    prices = np.arange(0, 39, 2)
    trade_list = []
    count = 0
    for i in tqdm(range(len(files[:days]))):
        book = get_raw_book(path+files[i])
        if not book.empty:
            trades = get_raw_trades(path_t+files_t[i])
            groups = pd.DataFrame(index=trades.index,
                                  columns=np.arange(0, len(prices)))
            count += 1
            # groups 0-9 Bid, 10-19 Ask
            for j in range(len(trades)):
                ind = book.index[book.index < trades.index[j]][-1]
                book_tmp = book.loc[ind, :].to_numpy()
                if trades.iloc[j, 0][0] == trades.iloc[j, 0][-1]:
                    vol = sum(trades.iloc[j, 1])
                    if np.mean(book_tmp[[0, 2]]) > trades.iloc[j, 0][0]:
                        groups.iloc[j, 0] = vol
                    else:
                        groups.iloc[j, 10] = vol
                elif trades.iloc[j, 0][0] > trades.iloc[j, 0][-1]:
                    # bid side
                    tmp_p = np.array(trades.iloc[j, 0])
                    tmp_v = np.array(trades.iloc[j, 1])
                    for num, pr in enumerate(np.unique(tmp_p)[:10]):
                        trade_ind = (tmp_p == pr)
                        vol = np.sum(tmp_v[trade_ind])
                        groups.iloc[j, num] = vol
                else:
                    # ask side
                    tmp_p = np.array(trades.iloc[j, 0])
                    tmp_v = np.array(trades.iloc[j, 1])
                    for num2, pr in enumerate(np.unique(tmp_p)[:10]):
                        trade_ind = (tmp_p == pr)
                        vol = np.sum(tmp_v[trade_ind])
                        groups.iloc[j, 10+num2] = vol
            trade_list.append(groups.fillna(0))
    output = pd.concat(trade_list)
    output['hour'] = output.index.hour
    output['mins'] = output.index.minute
    output['days'] = output.index.day
    return output

def summary_to_dist(trades, window, c_window, hour, min):
    data = trades.iloc[:,:22].groupby(['21','20']).sum()
    data = data.sum(axis=1).copy()
    data = data[(hour,min):(hour+window,min)].to_numpy()
    w = 2*c_window
    dist = []
    for i in range(len(data)//w):
        dist.append(np.sum(data[w*i:w*(i+1)]))
    return dist

def get_prob(df, count, hour, minute, window, level, vol):
    data = df.copy().to_numpy()
    filter_ = np.logical_and(data[:, 20] == hour, np.logical_and(
        data[:, 21] >= minute, data[:, 21] < minute+window))
    dist = data[filter_, level]
    return np.sum(dist >= vol)/count, np.sum(dist)/count

def new_prob(df, level, vol, hour, window):
    count = 0
    data = df.copy().to_numpy()
    def func(day):
        sub_count = np.zeros(10)
        for min_ in range(60-window):
            filter_ = np.logical_and(data[:,-1] == day, 
                np.logical_and(data[:, 21] == hour, 
                np.logical_and(data[:, 20] >= min_, data[:, 20] < min_+window)))
            sub_count += 1 * (np.sum(data[filter_,level:level+10], axis=0)>=vol)
        return sub_count/(60-window)
    out = Parallel(n_jobs=6)(delayed(func)(d) for d in np.unique(data[:,-1]))
    count = np.sum(np.array(out), axis=0)
    return count/len(np.unique(data[:,-1]))

def mean_exec(input: pd.DataFrame):
    timedelta = pd.Timedelta(seconds=60)
    data = input.copy()
    num_days = len(data.index.day.unique())
    data['minute'] = data.index.minute
    data['hour'] = data.index.hour
    data = data.groupby(['hour', 'minute']).sum()/num_days
    return data.groupby('hour').mean()


def plot_book(book_state: pd.Series):
    book_np = book_state.to_numpy()
    bid_vol = np.flip(book_np[np.arange(1, 40, 4)])
    bid_pr = np.flip(book_np[np.arange(0, 40, 4)])
    ask_vol = book_np[np.arange(3, 40, 4)]
    ask_pr = book_np[np.arange(2, 40, 4)]
    plt.bar(np.concatenate((bid_pr, ask_pr)),
            np.concatenate((bid_vol, ask_vol)))
    plt.show()


def transform_minutes(book_states):
    in_time = True
    timestamp = pd.Timestamp(
        2021, book_tmp.index[0].month, book_tmp.index[0].day, 8, 0)
    minutes = np.zeros((int(60*8.5), 40))
    ind = 0
    while in_time:
        tmp_time = timestamp + pd.DateOffset(minutes=1)
        minutes[ind, :] = book_states[(book_states.index > timestamp) & (
            book_states.index < tmp_time)].mean()
        timestamp = tmp_time
        ind += 1
        if tmp_time > pd.Timestamp(2021, book_tmp.index[0].month, book_tmp.index[0].day, 16, 29):
            in_time = False
    return minutes


def build_training(mins):
    window = 20
    label_range = 3
    x_vals = np.zeros(
        (mins.shape[0]-window-label_range, window, mins.shape[1]))
    y_vals = np.zeros((mins.shape[0]-window-label_range, 1))
    for i in range(1, x_vals.shape[0]):
        x_vals[i-1, :, :] = mins[i:i+window, :].copy() / mins[i-1:i-1 +
                                                              window, :].copy() - 1
        y_vals[i-1, :] = np.mean(mins[i+window+label_range-1, [0, 2]]
                                 ) > np.mean(mins[i+window-1, [0, 2]])
    x_vals[np.isnan(x_vals)] = 0
    return x_vals, y_vals


def check_time_window():
    book_test = get_raw_book(path_book+files[0])
    in_time = True
    timestamp = pd.Timestamp(2021, 1, 5, 8, 0)
    while in_time:
        tmp_time = timestamp + pd.DateOffset(minutes=1)
        plot_book(book_test[(book_test.index > timestamp)
                  & (book_test.index < tmp_time)].mean())
        timestamp = tmp_time
        if tmp_time > pd.Timestamp(2021, 1, 5, 8, 25):
            in_time = False


stock_list = ['Adidas', 'Allianz', 'BASF', 'Bayer', 'BMW', 'Continental',
              'Covestro', 'Covestro', 'Daimler', 'DeutscheBank', 'DeutscheBörse']
path_trades = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/trades/'
path_book = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'
dir_list_t = [unicodedata.normalize('NFC', f) for f in os.listdir(path_trades)]
dir_list_b = [unicodedata.normalize('NFC', f) for f in os.listdir(path_book)]

x_data = None
y_data = None
for stock in stock_list:
    files = get_file_names(stock, dir_list_b)
    for file in tqdm(files):
        book_tmp = get_raw_book(path_book+file)
        if not book_tmp.empty:
            minutes = transform_minutes(book_tmp)
            out = build_training(minutes)
            if x_data is None:
                x_data = out[0]
                y_data = out[1]
            else:
                x_data = np.concatenate((x_data, out[0]), axis=0)
                y_data = np.concatenate((y_data, out[1]), axis=0)


def everknowing_entity():
    midpoints = get_midpoints(path_book+files[3])
    in_time = True
    timestamp = pd.Timestamp(
        2021, midpoints.index[0].month, midpoints.index[0].day, 8, 0)
    labels = {'time': [], 'label': []}
    ind = 0
    while in_time:
        tmp_time = timestamp + pd.DateOffset(minutes=1)
        mid_start = midpoints[(midpoints.index > timestamp) & (
            midpoints.index < timestamp+pd.DateOffset(seconds=5))].mean()
        mid_end = midpoints[(midpoints.index > tmp_time) & (
            midpoints.index < tmp_time+pd.DateOffset(seconds=5))].mean()
        labels['label'].append(mid_end > mid_start)
        labels['time'].append(timestamp)
        timestamp = tmp_time
        ind += 1
        if tmp_time > pd.Timestamp(2021, book_tmp.index[0].month, book_tmp.index[0].day, 16, 29):
            in_time = False
    labels = pd.DataFrame(labels).reset_index('time')
    return labels


def data_sample(i):
    stock_list = ['Adidas', 'Allianz', 'BASF', 'Bayer', 'BMW', 'Continental',
                  'Covestro', 'Covestro', 'Daimler', 'DeutscheBank', 'DeutscheBörse']
    stock = stock_list[i]
    path_book = u'C:/Users/Lucas/Downloads/archive/_shared_storage/read_only/efn2_backtesting/book/'
    dir_list_b = [unicodedata.normalize('NFC', f)
                  for f in os.listdir(path_book)]
    files = get_file_names(stock, dir_list_b)
    for file in files:
        book_tmp = get_raw_book(path_book+file)
        if not book_tmp.empty:
            minutes = transform_minutes(book_tmp)
            out = build_training(minutes)
        else:
            out = (np.empty(1), np.empty(1))
    return out


pool_job = multiprocessing.Pool()
samples = pool_job.map(data_sample, range(len(stock_list)))


model = Sequential()
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.3, noise_shape=(None, 20, 64)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(l1=1e-8, l2=1e-8),
                bias_regularizer=keras.regularizers.L2(1e-8),
                activity_regularizer=keras.regularizers.L2(1e-8)))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(
    learning_rate=1e-5), metrics=tf.keras.metrics.BinaryAccuracy())

model.fit(x_data, y_data, batch_size=8, epochs=4, validation_split=0.1)

# last result acc=58, val_acc=60
# multi stock predictor?
