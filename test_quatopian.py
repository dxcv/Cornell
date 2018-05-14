# -*- coding: utf-8 -*-
from zipline.api import order, record, symbol


def initialize(context):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))

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

    print(perf_manual)
    if context.save2mysql == 0:
        df_valid.to_csv('../results/fof_region_10.csv')
    else:
        return 0