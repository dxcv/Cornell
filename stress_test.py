'''
mk stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path

# parameter initialization
return_period = 1
cov_period = 1 # return period
data_need = max(return_period,cov_period)
# required return parameter
required_return = 0.12
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
asset_data_files = ['nasdaq100','world_bond','us_bond','reits','emerg_equity',
                    'commodity','S&P500','Interest_rate'] # data_list
# Your data address
cwd = os.getcwd()
file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
res_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/res/'
model_name = 'stress_test_0.16'
stock_num = 6
save_file = model_name
save_address = res_address + save_file
weight_new =[0.166527,0.397769,0.022563,0.111279,0.227824,0.074039]
# def fix_fund_order(weights, p=0.05):
#     """
#     Args:
#         weights: Allocation weights
#         p: The minimum acceptable weight of an asset
#     Returns:
#         a new weight with all allocation weights > 5%.
#
#     """
#     if isinstance(weights, pd.Series) and len(weights) > 1:
#         weights = weights.sort_values()
#         if weights.min() <= p:
#             df1 = 1 - weights / p
#             df2 = weights.cumsum().shift()
#             df2.fillna(0)
#             mask = df2 >= df1
#             df = weights[mask] / (1 - df2[mask].iloc[0])
#         else:
#             df = weights
#         return df
#     return weights


if __name__ == "__main__":
    df = ut.data_processing(file_address,asset_data_files)
    print(df.to_csv(res_address+'whole_data.csv'))
    unit = np.full([len(df.index), 1, ], 1)[:, 0]
    df['rebalancing'] = pd.Series()
    df['stoploss'] = pd.Series()
    df['nav'] = pd.Series(unit, index=df.index)
    # TODO:fill the weight_new
    weight_new = pd.Series(weight_new,index = df.columns[:stock_num])
    max_new = []
    reb_index = 0
    df['rebalancing'].ix[reb_index] = 1
    for i in range(1,len(df.index)):
        print(i)
        if i<data_need:
            continue
        # print(weight_new)
        # record historic max price
        # if reb_index!=0:
        #     temp_max = np.amax(df.iloc[reb_index:i, :stock_num ],axis =0)
        #     max_new = pd.Series(temp_max,index = df.columns[:stock_num])

        # rebalance in fixed frequency
        # if np.mod(i-return_period,freq)==0:
        #     # The price data of etf uis from the second df.column
        #     weight_new = ut.mk(df.iloc[:i,:stock_num],
        #                        return_period, cov_period, required_return/trading_days)
        #     # weight_new = fix_fund_order(weight_new)
        #     # TODO:how to record weights?
        #     print(df.index[i-1])
        #     print(weight_new)
        #     print('*'*50)
        #     df['rebalancing'].ix[i-1] = 1
        #     reb_index = i-1

        # stoploss
        # if len(max_new) != 0:
        #     weight_new,flag = stoploss(df, stop_loss, i, max_new, weight_new)
        #     if flag == 1:
        #         print(df.index[i - 1])
        #         print('stoploss!: ')
        #         print(weight_new)
        #         print('-' * 50)
        #         df['stoploss'].ix[i - 1] = 1
        #         reb_index = i-1
        if len(weight_new) != 0:
            df = ut.record_return(df, stock_num, i, reb_index, weight_new, leverage, trading_days)
    perf = ut.comput_idicators(df,trading_days,required_return_real,save_file,save_address+'.csv')