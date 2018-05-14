
'''
mk stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path

# parameter initialization
return_period = 180
cov_period = 180 # return period
data_need = max(return_period,cov_period)
# required return parameter
required_return = 0.16
required_return_real = 0.1
# rebalancce frequency
freq = 63
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

data_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/monte_data/monte_'
model_name = 'mk_non_value_nonusequity_smallcapgrowth_short0.5_stoploss0.1_0.16_180'
stock_num = 6

save_file = 'final_monte_'
save_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/monte_res/'+save_file
N = 1000
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
    unit = pd.Series(np.full((len(weight_new.index), 1), 1)[:,0],index = weight_new.index)
    unit[weight_new[weight_new < 0].index]*=-1.0
    stop_info = (df[stock].ix[i-1]-max_new[stock])/max_new[stock]*unit[stock]
    if len(stop_info[stop_info<-re])!=0:
        weight_new[stop_info[stop_info < -re].index] = 0
        weight_new = weight_new / weight_new.sum()
        flag = 1
    return weight_new,flag


if __name__ == "__main__":
    s=0
    for j in range(351,N):
    # def sim_process(j):

        df = pd.read_csv(data_address+str(j)+'.csv')
        df =df.iloc[:,1:]
        df =df.reindex()
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
            # record historic max price
            if reb_index!=0:
                temp_max = np.amax(df.iloc[reb_index:i, :stock_num ],axis =0)
                max_new = pd.Series(temp_max,index = df.columns[:stock_num])

            # rebalance in fixed frequency
            if np.mod(i-return_period,freq)==0:
                # The price data of etf uis from the second df.column
                weight_new = ut.mk(df.iloc[:i,:stock_num],
                                   return_period, cov_period, required_return/trading_days)
                # weight_new = fix_fund_order(weight_new)
                # TODO:how to record weights?
                # print(df.index[i-1])
                # print(weight_new)
                # print('*'*50)
                df['rebalancing'].ix[i-1] = 1
                reb_index = i-1

            # stoploss
            if len(max_new) != 0:
                weight_new,flag = stoploss(df, stop_loss, i, max_new, weight_new)
                if flag == 1:
                    # print(df.index[i - 1])
                    # print('stoploss!: ')
                    # print(weight_new)
                    # print('-' * 50)
                    df['stoploss'].ix[i - 1] = 1
                    reb_index = i-1
            if len(weight_new) != 0:
                df = ut.record_return(df, stock_num, i, reb_index, weight_new, leverage, trading_days)
        perf = ut.monte_compute(df,trading_days,required_return_real,)
        if s==0:
            dfs = perf
        else:
            dfs = pd.concat([dfs,perf],axis =1)
        s+=1
        print(j)

        if np.mod(j,50)==0:
            temp = dfs.transpose()
            temp = temp.reindex()
            temp.to_csv(save_address+str(j)+'.csv')