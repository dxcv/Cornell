# -*- coding: utf-8 -*-
"""
@author: i-luoli
"""

import functools

import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from scipy.optimize import minimize
from zipline.api import (
    order_percent,
    order_target_percent,
    set_commission,
    commission,
    set_max_leverage,
    record,
    schedule_function,
    date_rules,
    get_datetime,
    time_rules,
    symbol,
)
from zipline.utils.strategy_utils import _to_sql

# 096001 大成标普500  T+7 2011-03 美股
# 050025 博时标普500交易型开放式指数证券投资基金联接基金	500	T+9	美股	2012.6
# 000071 华夏恒生etf  T+2 H股 2012-08
# 070031 嘉实全球房地产 2012-07
# 000342 嘉实新兴市场债 2015-11
# 110020 易方达沪深300交易型开放式指数发起式基金联接基金	10	T+3	A股	2009.9
# 沪深300改为博时增强型基金 050002  T+3
# 162711 广发中证500交易型开放式指数证券投资基金联接基金(LOF)	10	T+3		2009.11
# 中证500改为建信增强型基金 000478  T+1
# 110006 易方达货币市场基金A	0	T+1	货币	2005.1
# 161120 易方达中债新综合债券指数发起式基金（LOF）C	100	T+3	债	2012.11
# 000217 华安易富黄金交易型开放式证券投资基金联接基金C	1	T+3	黄金	2013.8


REBALANCE = 0
STOPLOSS = 1
drowdown_para = 250  # 最大回撤 控制时用到的历史时间
return_para = 60  # MV中预期收益计算用的历史时间
cov_para= 60   # MV中预期协方差矩阵计算用的历史时间
drawdown_parameter = [0.086,0.08,0.074,0.068,0.062,0.056,0.05,0.044,0.038,0.032]

def get_weight(context):
    weight = {}
    portfolio = context.portfolio
    for id in portfolio.positions:
                weight[id] = portfolio.positions[id].amount * \
                          portfolio.positions[id].last_sale_price / \
                          portfolio.portfolio_value

    return pd.Series(weight)


def compute_drawdown(nav, period=-1):
    """ 计算历史max{任意连续period天最大回撤}
    Args:
        nav: A pd.Seires of portfolio values, or a numpy.array
        period:  The specified period for max drawdown
    Returns: The maximum drawdown during the period (Max Drawdown以正数表示)
    求历史最大回撤
    >>> import numpy as np
    >>> arr = [1, 1.05, 1.1, 1, 1.2, 0.9, 0.8, 0.7, 1]
    >>> data = pd.Series(arr)
    >>> data.index = pd.period_range('20121121',periods=len(arr))
    >>> np.testing.assert_approx_equal(compute_drawdown(data), -(0.7-1.2)/1.2)

    求任意4天历史最大回撤
    >>> np.testing.assert_approx_equal(compute_drawdown(data, period=4),
    ... -(0.7-1.2)/1.2)

    求任意3天历史最大回撤
    >>> np.testing.assert_approx_equal(compute_drawdown(data, period=3),
    ... -(0.8-1.2)/1.2)

    求任意2天历史最大回撤
    >>> np.testing.assert_approx_equal(compute_drawdown(data, period=2),
    ... -(0.7-0.8)/0.8)
    """
    if isinstance(nav, pd.Series):
        nav = nav.values
    draw_down = 0

    peak = nav[0]
    if period == -1:
        max_draw_down = 0
        for idx, v in enumerate(nav):
            if v >= peak:
                peak = v
            draw_down = (v-peak)/peak
            if draw_down < max_draw_down:
                max_draw_down = draw_down
        return -max_draw_down
    else:
        i = 0
        for idx, v in enumerate(nav):
            if v >= peak:
                peak = v
                i = idx
            elif v < peak:
                dd = 1 - v / peak
                if dd > draw_down:
                    draw_down = dd
            if period > 2 and idx - period + 1 == i:
                peak = np.max(nav[i+1:idx])
                i += np.argmax(nav[i+1:idx])+1
            elif period == 2:
                peak = v
                if i == idx-1:
                    draw_down = 0
        return draw_down


def opt_progress_cvxopt(returns, r, contraint):
    """ 利用cvxopt包进行优化, 求最优边界解
    Args:
        returns: a pd.Series of returns
        contraint: The risk aversion coefficient
    Returns: a pd.Series of optimal weights

    无调仓的优化结果应该是在边界上面
    >>> from test.markowitz_test import normal_rets, markowitz_analytical_sol
    >>> cov = np.power(np.diag(np.array([0.10, 0.14])/np.sqrt(252)), 2)
    >>> rets = normal_rets(cov, 2, 100)
    >>> sigma = rets.cov()
    >>> mu = rets.mean()
    >>> w = opt_progress_cvxopt(rets, 0.5)
    >>> ret = np.dot(mu, w.T)
    >>> risk_opt = np.sqrt(np.dot(np.dot(w.T, sigma), w))
    >>> risk_closed = markowitz_analytical_sol(mu, np.linalg.inv(sigma), ret)
    >>> np.testing.assert_approx_equal(risk_closed, risk_opt)
    """
    sigma = returns.cov()

    n = len(r)
    # 协方差矩阵（优化目标中的2次项系数）
    H = 1 / contraint * matrix(sigma.values)

    # 优化目标中的1次项系数，目前为0
    f = matrix(-np.ones(n)*r)

    # 不等式约束————标的资产权重下限为0
    A2 = -np.eye(n)
    b2 = np.zeros(n)

    # 不等式约束————标的资产权重上限为1
    A3 = -np.eye(n)
    b3 = np.ones(n)
    # 不等式约束 汇总
    A = matrix(np.vstack((A2, A3)))
    b = matrix(np.append(b2, b3))


    # 等式约束————标的权重求和为1
    Aeq = matrix(np.ones(n), (1, n))
    beq = matrix(1.0)
    # 求解函数
    solvers.options['show_progress'] = False
    Sol = solvers.qp(H, f, A, b, Aeq, beq)

    weight = np.array(Sol['x'].T)[0]
    return weight


def matrix_corr_clean(returns, r):
    # 相关系数矩阵清洗
    sigma = returns.cov()
    n=len(r)
    # 样本变量比率 T/N
    T,n =returns.shape
    p=n*1.00/T
    # 对相关系数矩阵 特征值分解
    Emiprical_corr=np.corrcoef(returns.T)
    lamda,vector=np.linalg.eig(Emiprical_corr)

    # 1. eugebvakye clipping
    alpha=pow(1+np.sqrt(p),2)
    avg_lamda=np.mean(lamda)
    lamda[lamda<alpha]=avg_lamda
    Real_corr=np.dot(np.dot(vector,np.linalg.inv(np.diag(lamda))),np.linalg.inv(vector))

    # 2. power-law
#    alpha=0.5
#    idx = lamda.argsort()
#    lamda = lamda[idx]
#    vector = vector[:,idx]
#
#    for i in range(n):
#        lamda[i]=2*alpha-1+(1-alpha)*np.sqrt(n*1.00/(i+1))
#    Real_corr=np.dot(np.dot(vector,np.linalg.inv(np.diag(lamda))),np.linalg.inv(vector))
#
#    # 3. classical shrinkage
#    alpha=0.5
#    Real_corr=(1-alpha)*Emiprical_corr+alpha*np.ones((n,n))
#
#    # 4. Ledoit-Wolf
#    alpha=pow(1+np.sqrt(p),2)
#    avg_p=(np.sum(Emiprical_corr)-n)/(n*n-n)
#    avg_p_matrix=np.zeros((n,n))+np.diag(np.ones(n))+np.ones((n,n))*avg_p-np.diag(np.ones(n))*avg_p
#    Real_corr=(1-alpha)*Emiprical_corr+alpha*avg_p_matrix

    # 用修正的相关系数矩阵修正 协方差矩阵
    sigma_clean= sigma.copy()/Emiprical_corr*Real_corr

    return sigma_clean


def opt_progress_cvxopt_adjust(returns, r, constraint, weight_old, sell_fee, buy_fee):
    """ 利用cvxopt包进行优化, 求最优边界解, 加入调仓成本 （赎回和申购费用）
    Args:
        returns: a pd.Series of returns
        constrainst: The risk aversion coefficient
        weight_old: 上一阶段的组合权重
        buy_fee: 申购费用
        sell_fee: 赎回费用
    Returns: a pd.Series of optimal weights taking fees into account

    调仓的优化结果应该是在边界内部
    >>> from test.markowitz_test import normal_rets, markowitz_analytical_sol
    >>> cov = np.power(np.diag([0.10, 0.14]/np.sqrt(252)), 2)
    >>> rets = normal_rets(cov, 2, 100)
    >>> sigma = rets.cov()
    >>> mu = rets.mean()
    >>> w = opt_progress_cvxopt_adjust(rets, 0.5, [1,0], -0.006, -0.01)
    >>> ret = np.dot(mu, w.T)
    >>> risk_opt = np.sqrt(np.dot(np.dot(w.T, sigma), w))
    >>> risk_closed = markowitz_analytical_sol(mu, np.linalg.inv(sigma), ret)
    >>> risk_closed <= risk_opt
    True
    """

    sigma = returns.cov()
    n = len(r)

    sigma = matrix_corr_clean(returns, r)
    # 协方差矩阵（优化目标中的2次项系数）
    H = 1 / constraint * matrix(np.vstack((np.hstack((sigma.values, np.zeros((n, 2 * n)))),
                                           np.zeros((2*n, 3*n)))))

    # 优化目标中的1次项系数，目前为0
    f1 = r * np.ones(n)
    f2 = buy_fee * np.ones(n) # 买入 天化
    f3 = sell_fee * np.ones(n) # 卖出手续费

    f = -matrix(np.concatenate((f1, f2, f3),axis=0))
    # 不等式约束————标的资产权重下限为0
    A1 = -np.eye(3*n)
    b1 = np.zeros(3*n)

    # 不等式约束————标的资产权重上限为1
    A2 = np.hstack((-np.eye(n), np.zeros((n, 2*n))))
    b2 = np.ones(n)

    # 不等式约束————权重变动量
    A3 = np.hstack((np.eye(n), -np.eye(n), np.zeros((n, n))))
    b3 = np.ones(n) * weight_old

    A4 = np.hstack((-np.eye(n), np.zeros((n, n)), -np.eye(n)))
    b4 = - np.ones(n) * weight_old

    # 不等式约束 汇总
    A=matrix(np.vstack((A1, A2, A3, A4)))
    b=matrix(np.concatenate((b1, b2, b3, b4),axis=0))

    # 等式约束————标的权重求和为1
    Aeq = matrix(np.hstack((np.ones(n),np.zeros(2*n))), (1, 3*n))
    beq = matrix(1.0)

    # 求解函数
    solvers.options['show_progress'] = False
    sol = solvers.qp(H, f, A, b, Aeq, beq)
    weight = np.array(sol['x'].T)[0, :n]

    return weight


# 优化函数入口
def iter_opt(context, netvalue, Max_back, back_holding_period, return_period,
             cov_period,df_index):
    p_Return = pd.DataFrame(netvalue.pct_change()[1:-1])
    p_index = (~(p_Return.isnull().any())) & (~(p_Return == np.inf).any())

    p_Return_x = p_Return.ix[len(p_Return)-return_period:len(p_Return), p_index]
    p_Cov_x = p_Return.ix[len(p_Return)-cov_period:len(p_Return), p_index]

    # 用于计算历史 最大回撤
    netvalue_x = netvalue.ix[len(netvalue)-back_holding_period:len(netvalue), p_index]
    size = netvalue_x.shape
    for i in xrange(size[1]):
        # normalized the NAV
        netvalue_x.iloc[:, i] = netvalue_x.iloc[:, i]/netvalue_x.iloc[0, i]

    weight_temp = get_weight(context)
    weight_old = pd.Series(np.zeros(len(p_Return_x.columns)),
                           index=p_Return_x.columns)

    for i in weight_temp.index:
        if not i in weight_old.index:

            try:
                weight_old[symbol(df_index.loc[i.symbol][0])] = weight_temp[i]
            except:
                import pdb
                pdb.set_trace()

        else:
            weight_old[i] = weight_temp[i]

    r = p_Return_x.mean()

    # C的初始值
    C = 1000
    while True:
        print ".",
        res = opt_progress_cvxopt_adjust(p_Cov_x, r, C, weight_old, -0.006/60, -0.01/60)
        dd = pd.DataFrame(np.dot(netvalue_x, res))
        back = dd.apply(compute_drawdown, axis=0, args= (back_holding_period,))
        if back[0] - Max_back < 0.00005:
            break
        else:
            if C < 0.0001:
                break
            C *= 0.95
    print "."
    new_weight = pd.Series(res, index=netvalue_x.columns)
    return new_weight


def fix_fund_order(weights, p=0.05):
    """ 从最低权重的基金开始，如果一个基金的权重<5%, 我们会剔除这个基金，并把他的权重
    按其他基金的相对权重分配给其他的基金，直到所有的基金的权重都>5%.

    Args:
        weights: Allocation weights
        p: The minimum acceptable weight of an asset
    Returns:
        a new weight with all allocation weights > 5%.
    >>> import numpy as np
    >>> from random import random
    >>> arr = np.array([random() for i in range(10)])
    >>> rand_input = arr/sum(arr)
    >>> weights = pd.Series(rand_input)
    >>> new_weights = fix_fund_order(weights)
    >>> np.testing.assert_almost_equal(sum(new_weights),1)
    >>> (new_weights > 0.05).all()
    True
    """
    if isinstance(weights, pd.Series) and len(weights) > 1:
        weights = weights.sort_values()
        if weights.min() <= p:
            df1 = 1 - weights / p
            df2 = weights.cumsum().shift()
            df2.fillna(0)
            mask = df2 >= df1
            df = weights[mask] / (1 - df2[mask].iloc[0])
        else:
            df = weights
        return df
    return weights


# 开始优化
def initialize(context):
    set_commission(commission.PerDollar(cost=0.0055))
    # context.strategy_name = "FOF_{mdd}"
    context.mdd = context.namespace["mdd"]
    context.fof_code = context.namespace["fof_code"]
    context.save2mysql = context.namespace.get("save_method", 0)
    context.i = 0
    schedule_function(
        scheduled_rebalance,
        date_rules.month_start(days_offset= 0),
        time_rules.market_open()
    )
    context.init_value= pd.Series() # 记录自最近一次调仓以来，组合内各基金所达到的最大净值
    context.adjust_date= 0 # 记录距离最近一次季调的时间
    context.stop_date = 0 # 记录距离最近一次止损的时间
    # 基金池
    stock = [ '000478.OF', '050002.OF', '110006.OF',
         '161120.OF', '000217.OF',"501018.OF",
         '000071.OF','070031.OF','000342.OF','B00.IPE',
              "HSCI.HI", '037.CS',"em_bond","reit",
              "SPTAUUSDOZ.IDC",'096001.OF','SPX.GI','000905.SH']
    context.stocks = []
    for sym in stock:
        context.stocks.append(symbol(sym))
    #  指数与基金的对应关系 fixed
    index = ["HSCI.HI",'037.CS',"EM_BOND", "REIT",
             "SPTAUUSDOZ.IDC","000905.SH","B00.IPE","NDX.GI",'SPX.GI']
    index_etf = ["000071.OF","161120.OF","000342.OF",\
                 "070031.OF","000217.OF","000478.OF",\
                 "501018.OF","160213.OF",'096001.OF']
    context.df_index = pd.DataFrame(data = index_etf,index = index,)

def rebalance(context, data):

    stocks = context.stocks
    df_index = context.df_index
    hist = data.history(stocks,'close',drowdown_para, '1d')
    hist = hist.dropna(how = 'any',axis =1)
    for i in df_index.index:
        j = symbol(df_index.loc[i][0])
        if j in hist.columns:
            try:
                hist =hist.drop(symbol(i),1)
            except:
                pass
        else:
            continue
    print hist.iloc[-1,:]
    back_holding_period = drowdown_para
    # window length for expected return
    return_period = return_para
    # window length for covariance matrix, 不能超过有效数据长度，否则会出错
    cov_period= cov_para
    # get the stock list available for trading
    # stocks = context.get_universe(data)
    # calculate the optimized portfolio weight

    weight_new = iter_opt(context, hist, context.mdd,
                          back_holding_period,
                           return_period, cov_period,df_index)
    # 小与p=0.05的基金被筛除掉
    weight_new = fix_fund_order(weight_new)
    print '调仓了'
    print weight_new
    record(weights=weight_new)
    # 下单
    weight_old = get_weight(context)
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

    # 下单完成，记录最大净值变化
    context.init_value= pd.Series(np.zeros(weight_new.size), index=weight_new.keys())
    for stock in weight_new.keys():
        cost_price = hist[stock][-1]
        context.init_value[stock] = cost_price

def scheduled_rebalance(context, data):
    month = get_datetime().month
    if month in range(2, 12, 3):
        rebalance(context, data)
        record(rebalance_reason=REBALANCE)
        context.adjust_date=0

def stop_loss(context, stock_to_sell, data):

    # 止损：  将 stock 仓位降低为0，并调入货币基金
    # replace_stock： 调入的货币基金， 目前使用OF110006
    # stock: 将要进行止损的基金
    # flag:  判断 replace_stock 是否存在于原有仓位中
    # weight 原有仓位权重
    # stop_weight： 止损仓位，等于原始仓位重stock 的权重
    flag = False #初始值为货币基金不在持仓当中
    replace_stock = "110006.OF"

    weight = get_weight(context)
    stop_weight = weight[stock_to_sell]
    stocks = get_weight(context).keys()
    for stock in stocks:
        if stock.symbol == replace_stock:
            flag = True

    # 记录仓位及价格变动
    weight = weight.drop(stock_to_sell)
    if flag:
        weight[symbol(replace_stock)] = stop_weight + weight[symbol(
        replace_stock)]
    else:
        weight[symbol(replace_stock)] = stop_weight
    weight=weight/np.sum(weight)
    # 更新 调入货币基金 的成本价格
    context.init_value[symbol(replace_stock)] = data[symbol(replace_stock)]["price"]
    record(weights=weight)
    record(rebalance_reason=STOPLOSS)

     # 止损+调入货币基金下单
    order_target_percent(stock_to_sell, 0)
    order_percent(symbol(replace_stock), stop_weight)

def handle_data(context, data):
    context.save2mysql = 0
    context.i += 1
    print str(get_datetime())
    record(weights=None)
    # 按照最大回撤值止损
    context.adjust_date += 1
    context.stop_date += 1
    weight = get_weight(context)
    # print  weight
    stocks = weight.keys()
    stop_loss_flag = True # 记录本次是否

    # 更新最大净值
    for stock in stocks:
        current_price = data[stock]["price"]
        try:
            current_price > context.init_value[stock]
        except:
            assert(False)
        if current_price > context.init_value[stock]:
            context.init_value[stock] = current_price


    # 判断是否需要止损
    if context.adjust_date > 5 and context.adjust_date < 55 and \
            context.stop_date > 5:
        ben_loss = -0.05
        for stock in stocks:
            current_price = data[stock]["price"]
            returns = (current_price - context.init_value[stock])/context.init_value[stock]
            if returns < 0:
                # 分风险类别进行止损,激进组合止损不同
                if context.mdd < 0.07:
                    a = returns
                else:
                    a = returns * weight[stock] / (1 + returns)
                if a < ben_loss:
                    stop_loss(context, stock, data)
                    print "!!紧急调仓: " + str(get_datetime())
                    print stock,current_price,context.init_value[stock]
                    stop_loss_flag = False
            else:
                continue

    if stop_loss_flag:
        record(weights=None)
        record(rebalance_reason=None)
    else:
        context.stop_date = 0

    context.old_value = context.portfolio.portfolio_value

# 单日下跌超过5%，再平衡
#    if (context.portfolio.portfolio_value - context.old_value) / \
#            context.old_value < -0.05:
#        print str(get_datetime())
#        rebalance(context, data)
#    else:
#         record(weights=None)
#    context.old_value = context.portfolio.portfolio_value

def analyze(context, perf_manual):
    mdd = context.mdd
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
        # print "to_csv!!!"
        df_valid.to_csv('./results/fof_markowitz_0.05_test_'+str(
            mdd)+'.csv')
    else:
        _to_sql(df_valid, context.fof_code)
