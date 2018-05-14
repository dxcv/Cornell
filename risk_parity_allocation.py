'''
Risk-parity strategy
'''
import numpy as np
import pandas as pd
import utils as ut

# parameter initialization
LOOKBACK_PERIOD = 250 # return period
data_need = LOOKBACK_PERIOD
freq = 125
leverage = 1
trading_days = 250.0
required_return_real = 0.05
# put all the stock data in front of 'SP500'
asset_data_files = ['nasdaq100','svg_growth',
                    'world_bond','us_bond','reits','emerg_equity',
                    'commodity','S&P500','Interest_rate'] # data_list
# Your data address
file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
res_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/res/'
model_name = 'rp_125'
stock_num = 7
save_file = model_name
save_address = res_address + save_file

if __name__ == "__main__":
    df = ut.data_processing(file_address, asset_data_files)
    print(df.head())
    unit = np.full([len(df.index), 1, ], 1)[:, 0]
    df['rebalancing'] = pd.Series()
    df['stoploss'] = pd.Series()
    df['nav'] = pd.Series(unit, index=df.index)
    weight_new = []
    max_new = []
    reb_index = 0
    for i in range(LOOKBACK_PERIOD,len(df.index)):
        if i < data_need:
            continue
        # record max price
        if reb_index != 0:
            temp_max = np.amax(df.iloc[reb_index:i, :stock_num], axis=0)
            max_new = pd.Series(temp_max, index=df.columns[:stock_num])
        if np.mod(i-LOOKBACK_PERIOD,freq)==0:
            weight_new = ut.risk_parity(df.iloc[i-LOOKBACK_PERIOD:i
                                      ,:stock_num],LOOKBACK_PERIOD)
            # TODO:how to record weights
            print(df.index[i-1])
            print(weight_new)
            print('*'*100)
            df['rebalancing'].ix[i-1] = 1
            reb_index = i-1

            # stoploss
            # if len(max_new) != 0:
            #     weight_new, flag = stoploss(df, stop_loss, i, max_new, weight_new)
            #     if flag == 1:
            #         print(df.index[i - 1])
            #         print('stoploss!: ')
            #         print(weight_new)
            #         print('-' * 50)
            #         df['stoploss'].ix[i - 1] = 1
            #         reb_index = i - 1
        if len(weight_new) != 0:
            df = ut.record_return(df, stock_num, i, reb_index, weight_new, leverage, trading_days)
    perf = ut.comput_idicators(df, trading_days, required_return_real, save_file, save_address + '.csv')
