'''
mk stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path

# parameter initialization
return_period = 250
cov_period = 250 # return period
data_need = max(return_period,cov_period)
# required return parameter
required_return = 0.12
required_return_real = 0.09
# rebalancce frequency
freq = 250
# stoploss
stop_loss = 0.1
# calendar year
trading_days = 250.0
# no transaction fee
leverage = 0.95

# put all the stock data in front of 'SP500'
asset_data_files = ['nasdaq100','mk',
                    'S&P500','Interest_rate'] # data_list
# Your data address

file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
res_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/res/'
model_name = 'mk_employee_20_0.12_0.09_2'
stock_num = 2
save_file = model_name
save_address = res_address + save_file

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

def record_weights(df,i,weights):
    return_vector = np.log(df.iloc[i, :stock_num]) - np.log(df.iloc[i - 1, :stock_num])
    sum_return = np.dot(return_vector,weights.iloc[i-1,:].values)
    every_re = np.multiply(weights.iloc[i-1,:],(return_vector+1))
    weights.iloc[i,:] = every_re / (1 + sum_return)
    return weights

if __name__ == "__main__":
    df = ut.data_processing(file_address,asset_data_files)
    print(df.to_csv(res_address+'whole_data.csv'))
    unit = np.full((len(df.index), 1), 1)[:, 0]
    df['rebalancing'] = pd.Series()
    df['stoploss'] = pd.Series()
    df['nav'] = pd.Series(unit, index=df.index)
    weight_new =[]
    max_new = []
    unit = np.full((len(df.index),stock_num),np.nan)
    weights = pd.DataFrame(unit,index = df.index,columns = df.
                           columns[:stock_num])
    reb_index = 0
    for i in range(return_period,len(df.index)):
        if i<data_need:
            continue
        # record historic max price
        if reb_index!=0:
            temp_max = np.amax(df.iloc[reb_index:i, :stock_num ],axis =0)
            max_new = pd.Series(temp_max,index = df.columns[:stock_num])

        # stoploss
        if len(max_new) != 0:
            weight_new, flag = stoploss(df, stop_loss, i, max_new, weight_new)
            if flag == 1:
                print(df.index[i - 1])
                print('stoploss!: ')
                print(weight_new)
                weights.iloc[i - 1, :] = weight_new.values
                print('-' * 50)
                df['stoploss'].ix[i - 1] = 1
                reb_index = i - 1
        # rebalance in fixed frequency
        if np.mod(i-return_period,freq)==0:
            # The price data of etf uis from the second df.column
            weight_new = ut.mk(df.iloc[:i,:stock_num],
                               return_period, cov_period, required_return/trading_days)
            # weight_new = fix_fund_order(weight_new)
            # TODO:how to record weights?
            print(df.index[i-1])
            print(weight_new)
            print('*'*50)
            weights.iloc[i-1,:] = weight_new.values
            df['rebalancing'].ix[i-1] = 1
            reb_index = i-1

        if len(weight_new) != 0:
            df = ut.record_return(df, stock_num, i, reb_index, weight_new, leverage, trading_days)
            weights = record_weights(df, i, weights)

    perf = ut.comput_idicators(df,trading_days,required_return_real,save_file,save_address+'.csv')
    weights = weights.dropna(how ='any',axis =0 )
    weights.to_csv(res_address+'weights_'+model_name+'.csv')