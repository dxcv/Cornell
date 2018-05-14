'''
mk stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path

# parameter initialization
return_period = 120
cov_period = 120 # return period
back_holding_period = 250
back_max = 0.08
data_need = max(return_period,cov_period)
# required return parameter
required_return = 0.18
required_return_real = 0.1
# rebalancce frequency
freq = 60
# stoploss
stop_loss = 0.1
# calendar year
trading_days = 250.0
# no transaction fee
leverage = 0.95

# put all the stock data in front of 'SP500'
asset_data_files = ['nasdaq100','svg_growth',
                    'world_bond','us_bond','reits','emerg_equity',
                    'commodity','S&P500','Interest_rate'] # data_list
# Your data address
cwd = os.getcwd()
file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
res_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/res/'
model_name = 'mk_draw_non_value_nondevequity_smallcap_short0.5_stoploss0.1_0.08'
stock_num = 7
save_file = model_name
save_address = res_address + save_file


def stoploss(df,re,i,max_new,weight_new):
    flag = 0
    stock = weight_new[weight_new != 0].index.intersection(df.columns[:stock_num])
    # creat indicator for position
    unit = pd.Series(np.full([len(weight_new.index), 1, ], 1)[:, 0],index = weight_new.index)
    unit[weight_new[weight_new < 0].index]*=-1.0
    stop_info = (df[stock].ix[i-1]-max_new[stock])/max_new[stock]*unit[stock]
    if len(stop_info[stop_info<-re])!=0:
        weight_new[stop_info[stop_info < -re].index] = 0
        weight_new = weight_new / weight_new.sum()
        flag = 1
    return weight_new,flag

if __name__ == "__main__":
    df = ut.data_processing(file_address,asset_data_files)
    print(df.head())
    unit = np.full([len(df.index), 1, ], 1)[:, 0]
    df['rebalancing'] = pd.Series()
    df['stoploss'] = pd.Series()
    df['nav'] = pd.Series(unit, index=df.index)
    weight_new =[]
    max_new = []
    reb_index = 0
    for i in range(return_period,len(df.index)):
        if i<data_need:
            continue
        # record max price
        if reb_index!=0:
            temp_max = np.amax(df.iloc[reb_index:i, :stock_num ],axis =0)
            max_new = pd.Series(temp_max,index = df.columns[:stock_num])

        # rebalance in fixed frequency
        if np.mod(i-return_period,freq)==0:
            # The price data of etf uis from the second df.column
            weight_new = ut.mk_drawdown(df.iloc[:i, :stock_num], back_max,
                                        back_holding_period, return_period, cov_period)
            # weight_new = fix_fund_order(weight_new)
            # TODO:how to record weights?
            print(df.index[i-1])
            print(weight_new)
            print('*'*50)
            df['rebalancing'].ix[i-1] = 1
            reb_index = i-1

        # stoploss
        if len(max_new) != 0:
            weight_new,flag = stoploss(df, stop_loss, i, max_new, weight_new)
            if flag == 1:
                print(df.index[i - 1])
                print('stoploss!: ')
                print(weight_new)
                print('-' * 50)
                df['stoploss'].ix[i - 1] = 1
                reb_index = i-1
        if len(weight_new) != 0:
            df = ut.record_return(df, stock_num, i, reb_index, weight_new, leverage, trading_days)
    perf = ut.comput_idicators(df,trading_days,required_return_real,save_file,save_address+'.csv')