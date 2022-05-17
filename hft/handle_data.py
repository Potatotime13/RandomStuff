# create stock list
# iterate stocks
# iterate days
# read json to dict
# create day summary
# store day summayr in list
# save combined summary for stock

# importing the module
import json
import pandas as pd
import os

def get_file_names(stock, dir_list):
    tmp_dir_list = []
    for file_name in dir_list:
        if stock in file_name:
            tmp_dir_list.append(file_name)
    return tmp_dir_list

stock_list = ['Adidas', 'Allianz','BASF','Bayer','BMW','Continental','Covestro','Covestro','Daimler','DeutscheBank','DeutscheB├Ârse']
summary = {}
path = u'C:/Users/Lucas\Downloads/test_trade_download/trades'
dir_list = os.listdir(path)

for stock in stock_list:
    with open(r'C:\Users\Lucas\Downloads\test_trade_download\trades\Trades_'+stock+'_DE_20210105_20210105.json', 'r') as json_file:
        data = json.load(json_file)
    data_df = pd.DataFrame(data)
    data_df['TIMESTAMP_UTC'] = pd.to_datetime(data_df['TIMESTAMP_UTC'])
    data_df['hours'] = data_df['TIMESTAMP_UTC'].dt.hour
    data_df['Volume_agr'] = data_df.apply(lambda row : sum(row['Volume']), axis = 1)
    summary = data_df[['hours','Volume_agr']].groupby(['hours']).sum()