# create stock list
# iterate stocks
# iterate days
# read json to dict
# create day summary
# store day summayr in list
# save combined summary for stock

# importing the module
import json
import numpy as np
import pandas as pd
import os
import unicodedata
import matplotlib.pyplot as plt

def get_file_names(stock, dir_list):
    tmp_dir_list = []
    for file_name in dir_list:
        if stock in file_name:
            tmp_dir_list.append(file_name)
    return tmp_dir_list

stock_list = ['Adidas', 'Allianz','BASF','Bayer','BMW','Continental','Covestro','Covestro','Daimler','DeutscheBank','DeutscheBörse']
volume_summary = {}
path = u'C:/Users/Lucas/Downloads/test_trade_download/trades'
dir_list = [unicodedata.normalize('NFC', f) for f in os.listdir(path)]

for stock in stock_list:
    files = get_file_names(stock,dir_list)
    hour_avg = None
    for file in files:
        with open(u'C:/Users/Lucas/Downloads/test_trade_download/trades/'+file, 'r') as json_file:
            data = json.load(json_file)
        data_df = pd.DataFrame(data)
        data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])
        data_df['hours'] = data_df['TIMESTAMP_UTC'].dt.hour
        data_df['Volume_agr'] = data_df.apply(lambda row : sum(row['Volume']), axis = 1)
        summary = data_df[['hours','Volume_agr']].groupby(['hours']).sum()
        if hour_avg is None:
            hour_avg = summary['Volume_agr'].to_numpy()
        else:
            hour_avg += summary['Volume_agr'].to_numpy()
    volume_summary[stock] = (hour_avg/len(files)).tolist()

# TODO save as csv / maybe with minute window

def plot_vols(volume_summary:dict):
    fig = plt.figure()
    ax = plt.subplot(111)

    for stock in volume_summary:
        ax.plot(volume_summary[stock], label=stock)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylim([0,2])
    plt.xlabel('hours')
    plt.ylabel('avg vol')
    plt.show()