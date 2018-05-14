# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:07:36 2016

@author: i-mengqifei
"""

import pandas as pd
import numpy as np
import datetime
from zipline.api import (
    commission,
    set_commission,
    set_max_leverage,
    record,
    order_percent,
    order_target_percent,
    get_datetime,
    schedule_function,
    get_datetime,
    date_rules,
    time_rules,
    symbol,
    order_target_percent,

)


# 调仓理由
REBALANCE = "{date}, 根据240个交易日两种基金的波动率和协方差矩阵计算权重，重新调整。" \
# parameter
LOOKBACK_PERIOD = 240
T = 240
scheduled_month = 3

# 设置基金池
Emerging_market_bonds = "002401.OF"
America_stock = "096001.OF"
Hongkong_stock = "000071.OF"
gold = "000217.OF"
oil = "162411.OF"
house = "206011.OF"


def get_weight(context):
    weight = {}
    portfolio = context.portfolio
    for id in portfolio.positions:
                weight[id] = portfolio.positions[id].amount * \
                          portfolio.positions[id].last_sale_price / \
                          portfolio.portfolio_value
    return pd.Series(weight)


def patch_fund(context,data):

    stocks = context.stocks
    hist = data.history(stocks,'close',LOOKBACK_PERIOD,'1d')

    index = ["HSCI.HI","SPX.GI","EM_BOND","B00.IPE", "REIT",
             "SPTAUUSDOZ.IDC"]
    index_etf = ["000071.OF","096001.OF","002401.OF","162411.OF",\
                 "206011.OF","000217.OF"]

    df_index = pd.DataFrame(data = index_etf,index = index,)
    hist = hist.dropna(how = 'any',axis =1)
    for i in df_index.index:
        j = symbol(df_index.loc[i][0])
        if j in hist.columns:
        #避免有指数的数据不存在的情况，TODO：指数数据缺失？
            try:
                hist =hist.drop(symbol(i),1)
            except:
                pass
        else:
            continue

    return hist.columns


def initialize(context):
    # add_history(LOOKBACK_PERIOD, '1d', 'price')
    #set_max_leverage(1.25)
    context.i = 0
    set_commission(commission.PerDollar(cost=0.006))
    context.fof_code = context.namespace["fof_code"]
    context.save2mysql = context.namespace.get("save_method", 0)
    schedule_function(
        scheduled_rebalance,
        date_rules.month_start(days_offset= 0),
        time_rules.market_open()
    )
    context.weights = pd.Series()
    stock = [house,oil,Emerging_market_bonds,America_stock,Hongkong_stock,\
    gold,"HSCI.HI", "SPX.GI",'B00.IPE',"em_bond","reit",
              "SPTAUUSDOZ.IDC"]
    context.stocks = []
    for i in stock:
        context.stocks.append(symbol(i))


def scheduled_rebalance(context,data):
    month = get_datetime().month
    if month == scheduled_month:
        stocks = patch_fund(context,data)
        print (stocks)
        # print stocks
        rebalance(context,data,stocks)


def rebalance(context,data,stocks):

    hist = data.history(stocks,'close',LOOKBACK_PERIOD,'1d')
    N = len(stocks)

    X1 = np.array(hist.iloc[0:T-1,:])
    X2 = np.array(hist.iloc[1:T,:])
    ReturnMatrix = (X2-X1)/X1

    Cov_Var = np.cov(ReturnMatrix.T)
    Var = []
    for n in range(0,N):
        Var.append(np.var(ReturnMatrix[:,n]))

    StdVar =np.sqrt(Var)

    p = np.zeros((N,N))
    y = np.zeros((N+1,1))
    w = np.zeros((N,1))
    w_1 = np.zeros((N,1))
    w_2 = np.zeros((N,1))

    for m in range(0,N):
        for n in range(0,N):
            p[m,n] = Cov_Var[m,n]/StdVar[m]/StdVar[n]

    for n in range(0,N):
        y[n,0] = 0.5

    y[N,0] = 0.5
    for j in range(0,1000):
        for n in range(0,N):
            w[n,0] = y[n,0]
        lamda = y[N,0]
        for n in range(0,N):
            w_1[n,0] = 1/w[n,0]
            w_2[n,0] = 1/w[n,0]/w[n,0]

        F1 = np.dot(Cov_Var,w) - lamda*w_1
        S = np.zeros((1,1))
        F = np.concatenate((F1,S))
        F[N,0] =  0
        for n in range(0,N):
            F[N,0] = F[N,0] + w[n,0]

        F[N,0] = F[N,0] - 1
        D = np.zeros((N,N))
        for p in range(0,N):
            D[p,p] = w_2[p,0]

        J = Cov_Var+lamda*D
        S = -w_1

        J = np.hstack((J, S))
        S = np.ones((1, N+1))
        J = np.concatenate((J, S))
        c = w
        c = np.concatenate((c, np.zeros((1, 1))))
        c[N, 0] = lamda
        y = c - np.dot(np.linalg.inv(J), F)
        j = j+1
    W = np.zeros((N))
    for n in range(0, N):
        W[n] = y[n, 0]
    weight_new = pd.Series(W,index= hist.columns)
    print ("在{}调仓了！".format(str(get_datetime())))
    print (weight_new)
    record(weights = weight_new)
    # record(rebalance_reason=REBALANCE.format(**{"date":get_datetime().strftime(
    #     "%Y年%m月%d日")}))
    record(TRC=y[-1, 0])
    record(MRC=y[-1, 0] / y[:-1, 0])
    context.weights = weight_new

    # 下单
    weight_old = get_weight(context)
    # print weight_old
    for stock in weight_old.keys():
        if stock not in weight_new.keys():
            order_target_percent(stock, 0)

    change = {}
    for stock in weight_new.keys():
        if stock not in weight_old.keys():
            change[stock] = weight_new[stock]
        else:
            change[stock] = weight_new[stock] - weight_old[stock]

    for stock in sorted(change, key=change.get):
        order_percent(stock, change[stock])



def handle_data(context, data):

    context.i += 1
    print (get_datetime())

    record(weights=None)
    record(TRC=None)
    record(MRC=None)
    record(rebalance_reason = None)


def analyze(context, perf_manual):
    # mdd = context.mdd
    # calculate the return result
    # zipline的所有period_return都表示每一个时间点的整体收益R=(1+r1)...(1+rn)-1
    # TODO: 目前所有计算得自己做，zipline暂时还没有办法
    # 找到第一个交易的时间点
    # 更正benchmark return
    first_valid_idx = perf_manual['weights'].first_valid_index()
    # 计算需要的columns
    col = ['pnl', 'portfolio_value', 'benchmark_period_return',
                 'treasury_period_return','rebalance_reason','max_drawdown',
           'weights',\
          'gross_leverage','positions','orders']
    df_valid = perf_manual.ix[first_valid_idx:,col ]

    benchmark_R = df_valid['benchmark_period_return'] + 1
    df_valid.loc[:, 'benchmark_return'] = benchmark_R / benchmark_R.shift(1)
    df_valid.ix[0, 'benchmark_return'] = 1
    # 更正algorithm return
    df_valid.loc[:, 'algorithm_return'] = df_valid['pnl'] / df_valid[
        'portfolio_value'].shift(1) + 1
    df_valid.ix[0, 'algorithm_return'] = 1

    trading_days = 252.0

    def expanding_apply(df, fun, *args, **kwargs):
        if isinstance(df,np.ndarray):
            df = pd.DataFrame(df)
            _args = kwargs.pop("args", ())
            return np.array(df.expanding(*args, **kwargs).
                            apply(fun, args=_args).loc[:,0].values)
        _args = kwargs.pop("args", ())
        return df.expanding(*args, **kwargs).apply(fun, args=_args)

    # Volatility
    df_valid.loc[:, 'algo_volatility'] = expanding_apply(df_valid[
         'algorithm_return'] - 1, np.std) * np.sqrt(trading_days)
    # 夏普比率
    df_valid.loc[:, 'xret'] = (df_valid['algorithm_return'] - 1) - df_valid[
        'treasury_period_return'] / trading_days

    df_valid.loc[:, 'sharpe'] = expanding_apply(df_valid['xret'], lambda
        xret: np.sqrt(252) * xret.mean()/xret.std())

    # Dollar Growth
    df_valid.loc[:, 'dollar_growth'] = df_valid['algorithm_return'].cumprod()
    df_valid.loc[:,'benchmark_growth'] = df_valid['benchmark_return'].cumprod()

    # 年化收益 （几何增长率）
    df_valid.loc[:, 'cagr'] = expanding_apply(df_valid['dollar_growth'],
        lambda x: np.power(x[-1]/x[0], 1 / (len(x) / trading_days))- 1)

    # 战胜benchmark 的胜率
    wins = np.where(df_valid['algorithm_return'] >= df_valid[
        'benchmark_return'],  1.0, 0.0)
    df_valid.loc[:, 'hit_rate'] = wins.cumsum()/expanding_apply(wins, len)

    # 改变所有inf 到NA
    df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan

    # 统计短期收益率
    def cal_ret(x,p):
        xx = pd.Series(x)
        return xx.pct_change(p).median() * trading_days/p
    df_valid.loc[:,'month_one'] = expanding_apply(df_valid.loc[:,
                                                    'dollar_growth'],\
                            cal_ret,args = (20,))
    df_valid.loc[:,'month_3'] = expanding_apply(df_valid.loc[:,
                                                   'dollar_growth'],\
                            cal_ret,args = (60,))
    df_valid.loc[:,'month_12'] = expanding_apply(df_valid.loc[:,
                                                   'dollar_growth'],\
                            cal_ret,args = (252,))
    # 统计大于20%的概率
    def prob(x,p,pb):
        xx = pd.Series(x)
        x1 = xx.pct_change(p)
        x2 = x1[~x1.isnull()]
        if len(x2) > 0:
            # print x2[x2 >= pb]
            return len(x2[x2 >= pb]) * 1.0 / len(x2)
        else:
            return np.nan

    df_valid.loc[:,'pb_month_1'] = expanding_apply(df_valid.loc[:,
                                                    'dollar_growth'],\
                            prob,args = (20,0.2 * 20/trading_days,))
    df_valid.loc[:,'pb_month_3'] = expanding_apply(df_valid.loc[:,
                                                    'dollar_growth'],\
                            prob,args = (60,0.2 * 60/trading_days,))
    df_valid.loc[:,'pb_month_12'] = expanding_apply(df_valid.loc[:,
                                                    'dollar_growth'],\
                            prob,args = (252,0.2 * 252/trading_days,))

    # 识别不同时间段计算年化收益
    def cagr_period(a,start_date,end_date):
        def change(a,d,type):
            for i in range(0,len(a)):
                if str(a.index[i])[:10] < d :
                    continue
                else:
                    break
            if type == 0:
                return i
            else:
                return i - 1
        x = a.iloc[change(a,start_date,0):change(a,end_date,1)]
        return (x[-1] - x[0]) / x[0] * trading_days / len(x)
    df_valid.loc[:,'bull_return'] = np.nan
    df_valid.loc[:,'slowbear_return'] = np.nan
    df_valid.loc[:,'fastbear_return'] = np.nan
    df_valid.loc[:,'monkey_return'] = np.nan

    df_valid.ix[-1,'bull_return'] = cagr_period(df_valid.loc[:,
                                                 'dollar_growth'],
                                              '2014-08-01','2015-05-30')
    df_valid.ix[-1,'slowbear_return'] = cagr_period(df_valid.loc[:,
                                                  'dollar_growth'],
                                              '2013-01-01','2014-08-01')
    df_valid.ix[-1,'fastbear_return'] = cagr_period(df_valid.loc[:,
                                                  'dollar_growth'],
                                              '2015-06-01','2015-12-31')
    df_valid.ix[-1,'monkey_return'] = cagr_period(df_valid.loc[:,
                                                 'dollar_growth'],
                                              '2016-01-01','2017-02-15')

    if context.save2mysql == 0:
        df_valid.to_csv('../results/fof_region_10.csv')




