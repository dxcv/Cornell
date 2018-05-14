'''
mk stratgy
'''
import numpy as np
import pandas as pd
import utils as ut
import os.path
import scipy.stats as stats

# Your data address
file_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/data/'
out_address = 'D:/Work&Study/NYU/PythonScripts/Cornell/monte_data/'
stock_num = 6
save_file = 'monte_'
trading_days= 250
N = 1000
Year = 10
le_ngth = Year*trading_days

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
def de_W(m,n,chol,loc =0,sigma =1):
    ind_w = np.random.normal(loc,sigma,(m,n,))
    return np.matmul(ind_w,chol)

def GBM(rate,sigma,start,W):
    sim = []
    for i in range(0,len(W)):
        if i==0:
            sim.append(start*np.exp(rate+sigma*W[i]))
        else:
            sim.append(sim[i-1]*np.exp(rate+sigma*W[i]))
    return pd.Series(sim)

def VasM(speed,lm,start,sigma,W):
    sim = []
    for i in range(0, len(W)):
        if i == 0:
            sim.append(start+speed*(lm-start)+sigma*W[i])
        else:
            sim.append(sim[i - 1] +speed*(lm-start)+sigma*W[i])
    return pd.Series(sim)

df = pd.read_csv(out_address+'price.csv')
df = df.set_index('Date')
print(df.head())
if __name__ == "__main__":
    for j in range(0,N):


        price_start =df.iloc[-1,:]
        temp =df.iloc[:,:stock_num+1]
        re_turn = temp.pct_change()
        rate = re_turn.mean()
        sigma_eq = re_turn.std()
        # interest_rate
        temp = df['Interest_rate']
        x = temp.shift()
        y = temp.ix[1:]
        xx = x.dropna(how='any')
        yy = y.dropna(how='any')
        slope, intercept,_,_,_ = stats.linregress(xx, yy)
        speed = 1 - slope
        lm = intercept / speed
        res = (y - x * slope - intercept)/trading_days*1.0
        sigma = np.std(res)
        temp = re_turn.dropna(how='any',axis =0)
        res = pd.Series(res,index =temp.index)
        re_turn = pd.concat([temp,res],axis = 1)

        corr = re_turn.corr()
        chol = np.linalg.cholesky(corr)


        # generate W process
        W = de_W(le_ngth,stock_num+2,chol)
        # GBM
        sim_data = pd.DataFrame()
        for i in range(0,stock_num+1):
            temp = GBM(rate.ix[i],sigma_eq.ix[i],price_start.ix[i],W[:,i])
            if i==0:
               sim_data = temp
            else:
                sim_data = pd.concat([sim_data,temp],axis = 1)

        in_sim = pd.Series(VasM(speed, lm, price_start.ix[-1], sigma, W[:,-1]))
        sim_data = pd.concat([sim_data,in_sim],axis = 1)
        sim_data = sim_data.reindex(np.linspace(0,le_ngth-1,num =le_ngth))
        sim_data.columns = df.columns

        sim_data.to_csv(out_address+save_file+str(j)+'.csv')
        print(out_address+save_file+str(j))
        print('--' * 50)