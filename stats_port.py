'''
Risk-parity stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path

# parameter initialization
return_period = 120
cov_period = 120 # return period
data_need = max(return_period,cov_period)
# required return
required_return = 0.08
# rebalancce frequency
freq = 60
# calendar year
trading_days = 250.0
# no transaction fee
leverage = 0.9
asset_data_files = ['bond_index','equity_index','S&P500','Interest_rate'] # data_list
# Your data address

file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
res_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
model_name = 'mk'
stock_num = 2
save_file = model_name
save_address = res_address + save_file

if __name__ == "__main__":
    df = ut.data_processing(file_address,asset_data_files)
    df = df.pct_change()
    # df.to_csv(res_address+'index_price.csv')
    re_turn = df.iloc[:,0]*0.5+df.iloc[:,1]*0.5+1
    df['nav'] = re_turn.cumprod()
    df['nav'].to_csv(res_address+'index_price.csv')

    # print(df.mean()*250)
    # print(df.std()*np.sqrt(250))
    # print(df.corr())