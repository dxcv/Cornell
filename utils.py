
"""
Function list
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from cvxopt import solvers, matrix

# Function list******************************************************************************
# import data
def input_data(file_name):
    # df = pd.DataFrame()
    df = pd.read_csv(file_name)
    df.sort_values(by = ['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date',drop=True)
    # rename the price column by the data_list name.
    col_name = file_name[file_name.rfind('/')+1:-4]
    df.rename(columns={'Close': col_name}, inplace=True)
    # fill data from previous da
    df.ffill()
    # just dealing with some odd data
    # if col_name =='Interest_rate':
    #     df.loc[df[col_name]=='.',col_name] = np.nan
    return df[col_name].astype('float32')

def data_processing(file_address,data_list):
    # import data
    df = pd.DataFrame()
    s = 0
    for st in data_list:
        # print(input_data(file_address + st + '.csv'))
        if s==0:
            df = input_data(file_address+st+'.csv')
        else:
            df = pd.concat([df, input_data(file_address+st+'.csv')], axis = 1)
        s+=1
    df = df.dropna(axis=0, how='any')
    return df

def record_return(df,stock_num,i,reb_index,weight_new,leverage,trading_days=250.0):
    cum_return = np.dot(np.log(df.iloc[i, :stock_num]) - np.log(df.iloc[reb_index, :stock_num]),
                        weight_new.values)
    df['nav'].ix[i] = df['nav'].ix[reb_index] * (1 + cum_return * leverage + (1 - leverage)*(i-reb_index)
                                                 * df['Interest_rate'].ix[reb_index] / 100 / trading_days)
    return df

# computing indicators
def comput_idicators(df,trading_days,required,save_file,save_address, whole=1):
    # TODO:net_value has some problem.
    # columns needed
    col = ['index_price','Interest_rate','nav','rebalancing','stoploss']
    df_valid = df.ix[:,col]
    start_balance = df.index[df['rebalancing']==1][0]
    df_valid = df_valid[df_valid.index >= start_balance]

    # daily return
    df_valid['return'] = np.log(df['nav'])-np.log(df['nav'].shift(1))
    # benchmark_net_value
    df_valid['benchmark'] = df_valid['index_price']/df_valid['index_price'].ix[0]
    # benchmark_return
    df_valid['benchmark_return'] = (df_valid['benchmark']-
                                           df_valid['benchmark'].shift(1))/\
                                   df_valid['benchmark'].shift(1)
    # Annualized return
    df_valid['Annu_return'] = pd.expanding_mean(df_valid['return']) * trading_days
    # Volatility
    df_valid.loc[:, 'algo_volatility'] = pd.expanding_std(df_valid
                                                          ['return']) * np.sqrt(trading_days)
    df_valid.loc[:, 'xret'] = df_valid['return'] - df_valid[
        'Interest_rate'] / trading_days/100
    df_valid.loc[:,'ex_return'] = df_valid['return'] - df_valid['benchmark_return']
    def ratio(x):
        return np.nanmean(x)/np.nanstd(x)
    # sharpe ratio
    df_valid.loc[:, 'sharpe'] = pd.expanding_apply(df_valid['xret'], ratio)\
                                * np.sqrt(trading_days)
    # information ratio
    df_valid.loc[:, 'IR'] = pd.expanding_apply(df_valid['ex_return'], ratio)\
                                * np.sqrt(trading_days)
    # Sortino ratio
    def modify_ratio(x,re):
        re /= trading_days
        ret = np.nanmean(x)-re
        st_d = np.nansum(np.square(x[x < re]-re))/x[x < re].size
        return ret/np.sqrt(st_d)
    df_valid.loc[:, 'sortino'] = pd.expanding_apply(df_valid['return'], modify_ratio
                                                    ,args=(required,))* np.sqrt(trading_days)
    # Transfer infs to NA
    df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan
    df_valid.loc[np.isinf(df_valid.loc[:, 'IR']), 'IR'] = np.nan
    # hit_rate
    wins = np.where(df_valid['return'] >= df_valid[
        'benchmark_return'],  1.0, 0.0)
    df_valid.loc[:, 'hit_rate'] = wins.cumsum()/pd.expanding_apply(wins, len)
    # 95% VaR
    df_valid['VaR'] = -pd.expanding_quantile(df_valid['return'], 0.05)*\
                      np.sqrt(trading_days)
    # 95% CVaR
    df_valid['CVaR'] = -pd.expanding_apply(df_valid['return'],
                                          lambda x: x[x < np.nanpercentile(x, 5)].mean())\
                       * np.sqrt(trading_days)

    if whole ==1:
    # max_drawdown
        def exp_diff(x,type):
            if type == 'dollar':
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max()))
            else:
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max())/xx.max())
            return xret
    # dollar
    #     xret = exp_diff(df_valid['cum_profit'],'dollar')
    #     df_valid['max_drawdown_profit'] = abs(pd.expanding_min(xret))
    # percentage
        xret = exp_diff(df_valid['nav'], 'percentage')
        df_valid['max_drawdown_ret'] = abs(pd.expanding_min(xret))
    # max_drawdown_duration:
    # drawdown_enddate is the first time for restoring the max
        def drawdown_end(x,type):
                xret= exp_diff(x,type)
                minloc = xret[xret == xret.min()].index[0]
                x_sub = xret[xret.index > minloc]
            # if never recovering,then return nan
                try:
                    return x_sub[x_sub==0].index[0]
                except:
                    return np.nan
        def drawdown_start(x,type):
                xret = exp_diff(x, type)
                minloc = xret[xret == xret.min()].index[0]
                x_sub = xret[xret.index < minloc]
                try:
                    return x_sub[x_sub==0].index[-1]
                except:
                    return np.nan
        df_valid['max_drawdown_start'] = pd.Series()
        df_valid['max_drawdown_end'] = pd.Series()
        df_valid['max_drawdown_start'].ix[-1] = drawdown_start(
            df_valid['nav'],'percentage')
        df_valid['max_drawdown_end'].ix[-1] = drawdown_end(
            df_valid['nav'], 'percentage')
    df_valid.to_csv(save_address)
    # =====result visualization=====
    plt.figure(1)
    if whole==1:
        plt.subplot(224)
        plt.plot(df_valid['nav'],label = 'strategy')
        plt.plot(df_valid['benchmark'],label = 'S&P500')
    plt.xlabel('Date')
    plt.legend(loc=0, shadow=True)
    plt.ylabel('Nav')
    plt.title('Nav of '+ save_file +' & SP500')

    # plt.subplot(223)
    # plt.plot(df_valid['cum_profit'],label = 'strategy')
    # plt.xlabel('Date')
    # plt.ylabel('Cum_profit')
    # plt.title('Cum_profit of ' + save_file)

    plt.subplot(221)
    plt.plot(df_valid['return'], label='strategy')
    plt.xlabel('Date')
    plt.ylabel('Daily_return')
    plt.title('Daily Return of ' + save_file)

    plt.subplot(222)
    x_return = df_valid[df_valid['return'].notna()].loc[:,'return']
    y_return = df_valid[df_valid[
        'benchmark_return'].notna()].loc[:,'benchmark_return']
    mu = x_return.mean()
    sigma = x_return.std()
    mybins = np.linspace(mu-3*sigma,mu+3*sigma,100)
    count_x,_,_ = plt.hist(x_return,mybins,normed=1,alpha=0.5,label = 'strategy')
    count_y,_,_ = plt.hist(y_return,mybins,normed =1,alpha=0.5,label = 'S&P500')
    plt.ylabel('density')
    plt.xlabel('daily_return')
    plt.title('Histogram of Daily Return for ' +
              save_file+' & SP500')
    plt.grid(True)
    # add normal distribution line
    y = mlab.normpdf(mybins, mu, sigma)
    plt.plot(mybins, y, 'r--', linewidth = 1,label = 'Normal of strategy')
    plt.legend(loc=0, shadow=True)
    # plt.tight_layout()
    plt.show()
    return df_valid

def cut_position(df,signal_col,trading_days,save_file,save_address,long =1):
    if long==1:
        return comput_idicators(df[df[signal_col]>0],trading_days,save_file,save_address,whole=0)
    return comput_idicators(df[df[signal_col]<0],trading_days,save_file,save_address,whole=0)

def compute_drawdown(nav, period=-1):
    """
    Args:
        nav: A pd.Seires of portfolio values, or a numpy.array
        period:  The specified period for max drawdown
    Returns: The maximum drawdown during the period (Max Drawdown is positive here)
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

def matrix_corr_clean(returns, r):
        # clean cov_matrix
        sigma = returns.cov()
        n = len(r)
        # T/N
        T, n = returns.shape
        p = n * 1.00 / T
        # eigen_decomposition
        Emiprical_corr = np.corrcoef(returns.T)
        lamda, vector = np.linalg.eig(Emiprical_corr)

        # 1. eugebvakye clipping
        alpha = pow(1 + np.sqrt(p), 2)
        avg_lamda = np.mean(lamda)
        lamda[lamda < alpha] = avg_lamda
        Real_corr = np.dot(np.dot(vector, np.linalg.inv(np.diag(lamda))), np.linalg.inv(vector))

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

        # modify corr_matrix
        sigma_clean = sigma.copy() / Emiprical_corr * Real_corr

        return sigma_clean

# ================model==============================
# risk parity
def risk_parity(hist,T):

    N = len(hist.columns)

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

    W = np.zeros((N))
    for n in range(0, N):
        W[n] = y[n, 0]
    weight_new = pd.Series(W,index= hist.columns)
    return weight_new

def mk(netvalue,return_period,cov_period,required_return):
    """

    :param netvalue:
    :param back_holding_period:
    :param return_period:
    :param cov_period:
    :param required_return: daily return
    :return:
    """
    def opt_progress_cvxopt(returns, r, required_return):
        """
        Args:
            returns: a pd.Series of returns
            contraint: The risk aversion coefficient
        Returns: a pd.Series of optimal weights
       """
        sigma = matrix_corr_clean(returns, r)

        n = len(r)
        # Quadratic term
        H = 2 * matrix(sigma.values)

        # linear term
        f = matrix(-np.zeros(n) * r)

        # weights>0:no short
        A2 = -np.eye(n)
        b2 = np.ones(n)*0.5
        A4 = -matrix(np.ones(n) * r,(1,n))

        # weigths<1:no leverage
        A3 = np.eye(n)
        b3 = np.ones(n)

        # Sum of constraints
        A = matrix(np.vstack((A2, A3, A4)))
        b = matrix(list(b2)+list(b3)+[-required_return])

        # Sum of weights is one
        Aeq = matrix(np.ones(n), (1, n))
        beq = matrix(1.0)
        # function solver
        solvers.options['show_progress'] = False
        try:
            Sol = solvers.qp(H, f, A, b, Aeq, beq)
        except:
            return np.array(np.ones(n)/n*1.0)

        weight = np.array(Sol['x'].T)[0, :n]

        return weight

    p_Return = pd.DataFrame(netvalue.pct_change()[1:])
    p_index = (~(p_Return.isnull().any())) & (~(p_Return == np.inf).any())

    p_Return_x = p_Return.ix[(len(p_Return) - return_period+1):, p_index]
    p_Cov_x = p_Return.ix[(len(p_Return) - cov_period+1):, p_index]
    r = p_Return_x.median()

    res = opt_progress_cvxopt(p_Cov_x, r, required_return)
    new_weight = pd.Series(res, index=netvalue.columns)

    return new_weight

def mk_drawdown(netvalue, Max_back, back_holding_period, return_period,
                 cov_period,):

    def opt_progress_cvxopt(returns,r,C):

        sigma = matrix_corr_clean(returns, r)

        n = len(r)
        # Quadratic term
        H = 1/C * matrix(sigma.values)

        # linear term
        f = matrix(-np.ones(n) * r)

        # weights>0:no short
        A2 = -np.eye(n)
        b2 = np.zeros(n)
        # A4 = -matrix(np.ones(n) * r, (1, n))

        # weigths<1:no leverage
        A3 = np.eye(n)
        b3 = np.ones(n)

        # Sum of constraints
        A = matrix(np.vstack((A2, A3)))
        b = matrix(list(b2) + list(b3))

        # Sum of weights is one
        Aeq = matrix(np.ones(n), (1, n))
        beq = matrix(1.0)
        # function solver
        solvers.options['show_progress'] = False
        try:
            Sol = solvers.qp(H, f, A, b, Aeq, beq)
        except:
            return np.array(np.ones(n) / n * 1.0)

        weight = np.array(Sol['x'].T)[0, :n]

        return weight

    p_Return = pd.DataFrame(netvalue.pct_change()[1:])
    p_index = (~(p_Return.isnull().any())) & (~(p_Return == np.inf).any())

    p_Return_x = p_Return.ix[(len(p_Return) - return_period + 1):, p_index]
    p_Cov_x = p_Return.ix[(len(p_Return) - cov_period + 1):, p_index]

    # back_drawdown
    netvalue_x = netvalue.ix[len(netvalue) - back_holding_period:len(netvalue), p_index]
    size = netvalue_x.shape
    for i in xrange(size[1]):
        # normalized the NAV
        netvalue_x.iloc[:, i] = netvalue_x.iloc[:, i] / netvalue_x.iloc[0, i]

    r = p_Return_x.median()

    # initialzation of C
    C = 1000
    while True:
        print (".")
        res = opt_progress_cvxopt(p_Cov_x, r, C,)
        dd = pd.DataFrame(np.dot(netvalue_x, res))
        back = dd.apply(compute_drawdown, axis=0, args=(back_holding_period,))
        if back[0] - Max_back < 0.00005:
            break
        else:
            if C < 0.0001:
                break
            C *= 0.95
    print (".")
    new_weight = pd.Series(res, index=netvalue_x.columns)
    return new_weight

def monte_compute(df,trading_days,required,whole =1):
    col = ['S&P500', 'Interest_rate', 'nav', 'rebalancing',]
    df_valid = df.ix[:, col]
    start_balance = df.index[df['rebalancing'] == 1][0]
    df_valid = df_valid[df_valid.index >= start_balance]

    # daily return
    df_valid['return'] = np.log(df['nav']) - np.log(df['nav'].shift(1))
    # benchmark_net_value
    df_valid['benchmark'] = df_valid['S&P500'] / df_valid['S&P500'].iloc[0]
    # benchmark_return
    df_valid['benchmark_return'] = (df_valid['benchmark'] -
                                    df_valid['benchmark'].shift(1)) / \
                                   df_valid['benchmark'].shift(1)
    dff = pd.Series()
    # Annualized return
    dff['Annu_return'] = np.mean(df_valid['return']) * trading_days
    # Volatility
    dff['algo_volatility'] = np.std(df_valid['return']) * np.sqrt(trading_days)

    df_valid.loc[:, 'xret'] = df_valid['return'] - df_valid[
                                                       'Interest_rate'] / trading_days / 100
    df_valid.loc[:, 'ex_return'] = df_valid['return'] - df_valid['benchmark_return']

    def ratio(x):
        return np.nanmean(x) / np.nanstd(x)

    # sharpe ratio
    dff['sharpe'] = ratio(df_valid['xret'])* np.sqrt(trading_days)
    # information ratio
    dff['IR'] = ratio(df_valid['ex_return'])* np.sqrt(trading_days)
    # hit_rate
    wins = np.where(df_valid['return'] >= df_valid[
        'benchmark_return'], 1.0, 0.0)
    dff['hit_rate'] = wins.sum() / len(wins)
    # Sortino ratio
    def modify_ratio(x, re):
        re /= trading_days
        ret = np.nanmean(x) - re
        st_d = np.nansum(np.square(x[x < re] - re)) / x[x < re].size
        return ret / np.sqrt(st_d)
    def downside_risk(x):
        re = np.nanmean(x)
        st_d = np.nansum(np.square(x[x < re] - re))
        return np.sqrt(st_d)

    dff['sortino'] = modify_ratio(df_valid['return'],required) * np.sqrt(trading_days)
    dff['downside_risk'] = downside_risk(df_valid['return']) * np.sqrt(trading_days)
    # Transfer infs to NA
    # df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan
    # df_valid.loc[np.isinf(df_valid.loc[:, 'IR']), 'IR'] = np.nan
    # hit_rate
    # wins = np.where(df_valid['return'] >= df_valid[
    #     'benchmark_return'], 1.0, 0.0)
    # df_valid.loc[:, 'hit_rate'] = wins.cumsum() / pd.expanding_apply(wins, len)
    # 95% VaR
    dff['VaR'] = -df_valid['return'].quantile(q=0.05) *np.sqrt(trading_days)
    # 95% CVaR
    x = df_valid['return']
    dff['CVaR'] = -x[x < np.nanpercentile(x, 5)].mean()* np.sqrt(trading_days)

    if whole == 1:
        # max_drawdown
        def exp_diff(x, type):
            if type == 'dollar':
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max()))
            else:
                xret = pd.expanding_apply(x, lambda xx:
                (xx[-1] - xx.max()) / xx.max())
            return xret
            # dollar
            #     xret = exp_diff(df_valid['cum_profit'],'dollar')
            #     df_valid['max_drawdown_profit'] = abs(pd.expanding_min(xret))
            # percentage

        xret = exp_diff(df_valid['nav'], 'percentage')
        dff['max_drawdown_ret'] = abs(np.min(xret))

        # # max_drawdown_duration:
        # # drawdown_enddate is the first time for restoring the max
        # def drawdown_end(x, type):
        #     xret = exp_diff(x, type)
        #     minloc = xret[xret == xret.min()].index[0]
        #     x_sub = xret[xret.index > minloc]
        #     # if never recovering,then return nan
        #     try:
        #         return x_sub[x_sub == 0].index[0]
        #     except:
        #         return np.nan
        #
        # def drawdown_start(x, type):
        #     xret = exp_diff(x, type)
        #     minloc = xret[xret == xret.min()].index[0]
        #     x_sub = xret[xret.index < minloc]
        #     try:
        #         return x_sub[x_sub == 0].index[-1]
        #     except:
        #         return np.nan
        #
        # df_valid['max_drawdown_start'] = pd.Series()
        # df_valid['max_drawdown_end'] = pd.Series()
        # df_valid['max_drawdown_start'].ix[-1] = drawdown_start(
        #     df_valid['nav'], 'percentage')
        # df_valid['max_drawdown_end'].ix[-1] = drawdown_end(
        #     df_valid['nav'], 'percentage')

    return dff