# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from lib1 import featurelib
from zipline.pipeline.factors import CustomFactor
from zipline.api import (
    set_symbol_lookup_date,
    order_target,
    order_percent,
    get_open_orders,
    order_target_percent,
    symbols,
    schedule_function,
    date_rules,
    time_rules,
    commission,
    record,
    symbol,
    set_max_leverage,
    set_commission,
    set_slippage,
    slippage,
    attach_pipeline,
    pipeline_output,
    get_datetime,
)

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import CustomFilter,SpecificAssets
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import Returns, MaxDrawdown, SimpleMovingAverage, Latest
from empyrical import information_ratio, beta, max_drawdown
from sqlalchemy import create_engine
from sklearn import linear_model, ensemble

db = {
    "user": "chengxu",
    "host": "172.16.3.174",
    "port": "3306",
    "passwd": "chengxu.gbh",
    "db": "fof",
}

con = create_engine( 'mysql+mysqldb://{user}:{passwd}@{host}:{port}/{db}'.format(**db))
df = pd.read_sql_table("mutual_fund_classification", con)

rets = Returns(inputs=[USEquityPricing.close], window_length=2)

class Down_sharpe(CustomFactor):
    inputs=[rets]

    def __new__(cls, *args, **kwargs):
        self = super(Down_sharpe, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.rf =kwargs["rf"]
        self.c =kwargs["c"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            valid_r=pd.DataFrame(x)
            try:
                _dshp = valid_r.apply(featurelib.down_sharpe, args=(self.c, self.rf/self.c,), axis=0)
            except:
                _dshp = None
            out[:] =  _dshp
        else:
            out[:]= np.nan

class Drawdown(CustomFactor):
    inputs=[rets]

    def __new__(cls, *args, **kwargs):
        self = super(Drawdown, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            _drwd = nav.apply(max_drawdown)
            out[:] =  _drwd
        else:
            out[:]= np.nan

class Down_variation(CustomFactor):
    inputs=[rets]

    def __new__(cls, *args, **kwargs):
        self = super(Down_variation, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.c =kwargs["c"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            valid_r=pd.DataFrame(x)
            _dvar = valid_r.apply(featurelib.downside_var, args=(self.c,), axis=0)
            out[:] =  _dvar
        else:
            out[:]= np.nan

class Average_return(CustomFactor):
    inputs=[rets]

    def __new__(cls, *args, **kwargs):
        self = super(Average_return, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            valid_r=pd.DataFrame(x)
            mean_r= valid_r.mean()
            out[:] = mean_r
        else:
            out[:]= np.nan

class Lasting(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):
        self = super(Lasting, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.percent =kwargs["percent"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            abs_r = nav.pct_change(self.time)
            def percentile(x, q):
                valid_x = x[~np.isnan(x)]
                h = np.nanpercentile(valid_x, 100 - q, interpolation='higher')
                x[x >= h] = 1
                x[x < h] = 0
                return x

            threshold = abs_r.apply(percentile, raw=True, args=(self.percent,), axis=1)
            _lasting = threshold.apply(featurelib.lasting)
            out[:] = _lasting
        else:
            out[:]= np.nan

class Emr(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):
        self = super(Emr, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            y = nav.pct_change(self.time)
            _emr = pd.Series(np.zeros(y.shape[1])).astype(float)

            for j in range(0,y.shape[1]):
                temp = y.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    _emr[y.columns[j]] = np.nan
                else:
                    yy = np.array(temp[~temp.isnull()])
                    try:
                        _emr[y.columns[j]] = featurelib.emr(yy)
                    except :
                        _emr[y.columns[j]] = None
            out[:] = _emr
        else:
            out[:]= np.nan

class Negative_variation(CustomFactor):
    inputs=[rets]

    def __new__(cls, *args, **kwargs):
        self = super(Negative_variation, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.c =kwargs["c"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        # negative_variation
        self.i=self.i+1
        if self.i % 21  == 1 :
            valid_r=pd.DataFrame(x)
            benchmark = 0.0
            valid_stand_r = valid_r - benchmark
            _std_var = valid_stand_r.apply(featurelib.std_var, args = (self.c,),axis = 0)
            out[:] = _std_var
        else:
            out[:]= np.nan

class Mcv_in(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):
        self = super(Mcv_in, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.index_sid=kwargs["index_sid"]
        self.rf =kwargs["rf"]
        self.c =kwargs["c"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            y = nav.pct_change(self.time)
            ben_nav = pd.DataFrame(x[:,assets==self.index_sid])
            x1 = ben_nav.pct_change(self.time) - 0.025*5.0/252
            _mcv_in = pd.Series(np.zeros(y.shape[1])).astype(float)
            for j in range(0,y.shape[1]):
                temp = y.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    _mcv_in[y.columns[j]] = np.nan
                else:
                    yy = temp[~temp.isnull()]
                    xx = x1.iloc[yy.index,0]
                    _mcv_in[y.columns[j]] = featurelib._mcv(yy,xx,self.rf,self.c)
            out[:] = _mcv_in
        else:
            out[:]= np.nan

class Rank_stable(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):

        self = super(Rank_stable, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time=kwargs["time"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            abs_r= nav.pct_change(self.time)
            rank = abs_r.rank(axis=1, ascending=False)
            rank_score = rank.apply(featurelib.rank_score, axis=1, raw=True)
            _stable = rank_score.apply(featurelib.score_downstd,axis = 0)
            out[:] = _stable
        else:
            out[:]= np.nan

class CL(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):

        self = super(CL, cls).__new__(
            cls,
            outputs=["select_time","select_stock"],
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time=kwargs["time"]
        self.index_sid=kwargs["index_sid"]
        self.rf =kwargs["rf"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            ben_nav = pd.DataFrame(x[:,assets==self.index_sid])
            valid_regression_r1 = ben_nav.pct_change(self.time) - 0.025*5.0/252

            temp =  np.zeros(shape=(valid_regression_r1.shape))
            valid_regression_r2 = pd.DataFrame(temp,index = valid_regression_r1.index)
            x_matrix = pd.concat([valid_regression_r1,valid_regression_r2],axis = 1)

            cl_x1 = x_matrix.min(axis =1)
            cl_x2 = x_matrix.max(axis = 1)
            x1 = cl_x1
            x2 = cl_x2
            y = nav.pct_change(self.time)
            inter_index = y.index.intersection(x1.index.intersection(x2.index))
            y = y.loc[inter_index]
            x1 = x1.loc[inter_index]
            x2 = x2.loc[inter_index]
            _select_time =  pd.Series(np.zeros(y.shape[1])).astype(float)
            _select_stock =  pd.Series(np.zeros(y.shape[1])).astype(float)
            for j in range(0,y.shape[1]):
                temp = y.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    _select_time[y.columns[j]]  = np.nan
                    _select_stock[y.columns[j]] = np.nan
                else:
                    yy = temp[~temp.isnull()]
                    xx1 = x1[yy.index]
                    xx2 = x2[yy.index]
                    _select_time[y.columns[j]], _select_stock[y.columns[j]]  = featurelib._cl(yy,xx1,xx2,self.rf,)

            out["select_time"][:]= _select_time
            out["select_stock"][:]= _select_stock
        else:
            out[:]= np.nan

class Hit_rate(CustomFactor):
    inputs=[USEquityPricing.close]
    def __new__(cls, *args, **kwargs):
        self = super(Hit_rate, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time_window=kwargs["time_window"]
        self.time=kwargs["time"]
        self.index_sid=kwargs["index_sid"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            ben_nav = pd.DataFrame(x[:,assets==self.index_sid])
            valid_ben_rhorizon_all = ben_nav.pct_change(self.time)
            valid_rhorizon = nav.pct_change(self.time)
            def minus(a,b):
                return a-b
            valid_related_rhorizon = valid_rhorizon.apply(minus,args = (valid_ben_rhorizon_all.iloc[:,0],),axis =0)
            r = valid_related_rhorizon.iloc[-self.time_window-1:-1]
            _hit_rate = r.apply(featurelib.hit_rate, axis = 0)
            out[:]= _hit_rate
        else:
            out[:]= np.nan

class Value_at_risk(CustomFactor):
    inputs=[USEquityPricing.close]

    def __new__(cls, *args, **kwargs):

        self = super(Value_at_risk, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.q_value =kwargs["q_value"]
        self.time =kwargs["time"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            valid_r = nav.pct_change(self.time)
            _Var = pd.Series(np.zeros(valid_r.shape[1])).astype(float)

            for j in range(0,valid_r.shape[1]):
                temp = valid_r.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    _Var[valid_r.columns[j]] = np.nan
                else:
                    y = temp[~temp.isnull()]
                    try:
                        _Var[valid_r.columns[j]] = featurelib.std_rtn(y,self.q_value)
                    except:
                        _Var[valid_r.columns[j]]= np.nan
            out[:]= _Var
        else:
            out[:]= np.nan

class Beta_in(CustomFactor):
    inputs=[USEquityPricing.close]
    def __new__(cls, *args, **kwargs):

        self = super(Beta_in, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.index_sid=kwargs["index_sid"]
        self.rf =kwargs["rf"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            y = nav.pct_change(self.time)
            ben_nav = pd.DataFrame(x[:,assets==self.index_sid])
            valid_regression_r1 = ben_nav.pct_change(self.time) - 0.025*5.0/25
            x1 = valid_regression_r1.loc[y.index]
            wbeta = pd.Series(np.zeros(y.shape[1])).astype(float)

            for j in range(0,y.shape[1]):
                temp = y.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    wbeta[y.columns[j]] = np.nan
                else:
                    yy = temp[~temp.isnull()]
                    xx = x1.iloc[yy.index,0]
                    wbeta[y.columns[j]] =  beta(yy, xx, risk_free=self.rf)
            out[:]= wbeta
        else:
            out[:]= np.nan

class Bias_in(CustomFactor):
    inputs=[USEquityPricing.close]
    def __new__(cls, *args, **kwargs):

        self = super(Bias_in, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            _ma = nav.rolling(window=self.time,center=False).mean()
            _temp = nav/_ma -1
            _bias = _temp.iloc[-1,:]
            out[:]= _bias
        else:
            out[:]= np.nan

class Information_ratio(CustomFactor):
    inputs=[USEquityPricing.close]
    def __new__(cls, *args, **kwargs):

        self = super(Information_ratio, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.time =kwargs["time"]
        self.index_sid=kwargs["index_sid"]
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            nav=pd.DataFrame(x)
            valid_r = nav.pct_change(self.time)
            ben_nav = pd.DataFrame(x[:,assets==self.index_sid])
            valid_ben_rday=ben_nav.pct_change(self.time)
            ben_r = valid_ben_rday.loc[valid_r.index]
            _ir = pd.Series(np.zeros(valid_r.shape[1])).astype(float)
            for j in range(0,valid_r.shape[1]):
                temp = valid_r.iloc[:,j]
                if np.isnan(temp[self.time:]).any():
                    _ir[valid_r.columns[j]] = np.nan
                else:
                    yy = temp[~temp.isnull()]
                    xx = ben_r.iloc[yy.index,0]
                    _ir[valid_r.columns[j]] = information_ratio(yy,xx)
            out[:]= _ir
        else:
            out[:]= np.nan

class Lag_Return(CustomFactor):
    inputs=[USEquityPricing.close]
    def __new__(cls, *args, **kwargs):

        self = super(Lag_Return, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1 :
            prices=pd.DataFrame(x)
            lag_ret = (prices.iloc[-1,:] - prices.iloc[0,:]) / prices.iloc[0,:]
            out[:]= lag_ret
        else:
            out[:]= np.nan

class Lag_Sharpe(CustomFactor):
    inputs=[rets]
    def __new__(cls, *args, **kwargs):

        self = super(Lag_Sharpe, cls).__new__(
            cls,
            mask= kwargs["mask"],
            window_length = kwargs["window_length"]
        )
        self.i=0
        return self

    def compute(self, today, assets, out, x):
        ################
        self.i=self.i+1
        if self.i % 21  == 1:
            valid_r=pd.DataFrame(x)
            def sharpe(r):
                trading_days = 252.0
                rf= 0.025
                # 夏普比率
                adj_r = r - rf / trading_days
                sp = np.sqrt(trading_days) * adj_r.mean() / r.std()
                # 如果出现极大值，很可能是净值出现问题，此处将其设为nan,后续会删掉
                if (sp > 10000) | (sp< -10000):
                    sp = np.nan
                if np.isnan(r).any():
                    sp = np.nan
                return sp
            _dshp = valid_r.apply(sharpe, axis=0)
            out[:] =  _dshp
        else:
            out[:]= np.nan

def get_weight(context):
    weight_old = pd.Series()
    for stock in context.portfolio.positions.keys():
            weight_old[stock] = context.portfolio.positions[stock].amount * \
                context.portfolio.positions[stock].last_sale_price/ context.portfolio.portfolio_value
    return weight_old

def initialize(context):

    context.record_rank = {}
    context.save2mysql = 0
    # 模型训练参数
    context.horizon = 6
    context.percent = 5
    context.model_name = 'adaBoost'
    context.rolling = 1 # 是否为滚动，0为不滚动
    context.ishistory = 0 #是否用历史分类。0为不使用
    context.train_period = 12 * 1
    context.i = 0
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerDollar(cost=0.00325))

    month = 21
    week = 5
    rf = 0.025
    c = 252

    if context.namespace["fund_type"] == 'stock':
        benchmark = '000300.SH'
        fund_type = 1
    elif context.namespace["fund_type"] == 'hybrid':
        benchmark = '000300.SH'
        fund_type = 2
    elif context.namespace["fund_type"] == 'bond':
        benchmark = '037.CS'
        fund_type = 3
    # 选择基金池和benchmark
    ben_sid=symbol(benchmark).sid
    df_stock = df.query("update_time == update_time.max() and type_code=={"\
                        "type}".format(type = fund_type))
    # 基金900 开头的不需要，专门去掉
    df_stock = df_stock[df_stock["fund_code"]<'900000'].fund_code + ".OF"
    sfilt = SpecificAssets(symbols(benchmark,*tuple(df_stock))) # 使用到的数据，包括基金和对应benchmark指数
    sample_filt = SpecificAssets(symbols(*tuple(df_stock))) # 只包含基金，因为评级不应该包含benchmark，故在事后screen去掉benchmark

    # 只包含基金，因为评级不应该包含benchmark，故在事后screen去掉benchmark
    ## 16个因子指标
    down_sharpe = Down_sharpe(window_length =  9*month, rf=rf, c=c, mask=sfilt)
    drawdown = Drawdown(window_length =  9*month, mask=sfilt)
    dvar = Down_variation(window_length =  9*month, rf=rf, c=c, mask=sfilt)
    mean_r = Average_return(window_length =  9*month, mask=sfilt)
    lasting=Lasting(window_length =  9*month, mask=sfilt, time=6*month, percent = 20)
    emr=Emr(window_length =  9*month, mask=sfilt, time=6*month)
    rlt_var=Negative_variation(window_length =  9*month, c=c, mask=sfilt)
    mcv_in= Mcv_in(window_length =  9*month, rf=rf, c=c, mask=sfilt, time=week, index_sid=ben_sid)
    rank_stable=Rank_stable(window_length =  9*month, mask=sfilt, time=6*month)
    select_time, select_stock=CL(window_length =  9*month, rf=rf, mask=sfilt, time=week, index_sid=ben_sid)
    hit_rate=Hit_rate(window_length =  10*month, mask=sfilt, time=6*month, time_window= 9*month, index_sid=ben_sid)
    value_at_risk =Value_at_risk(window_length =  9*month, mask=sfilt, q_value = 5, time=6*month)
    beta_in =Beta_in(window_length =  9*month, mask=sfilt, rf=rf, time=week, index_sid=ben_sid)
    bias_in =Bias_in(window_length =  9*month, mask=sfilt, time =126)
    ir=Information_ratio(window_length =  9*month, mask=sfilt, time=1,index_sid=ben_sid)
    # 预测因变量Y
    _ry = Lag_Return(window_length =  context.horizon*month, mask=sfilt)
    _sp = Lag_Sharpe(window_length =  context.horizon*month, mask=sfilt)

    pipe = Pipeline(
        columns={
            "dhsp":down_sharpe,
            "drwd":drawdown,
            "dvar":dvar,
            "mean_r":mean_r,
            "lasting":lasting,
            "emr":emr,
            "rlt_var":rlt_var,
            "mcv_in":mcv_in,
            "rank_stable":rank_stable,
            "select_time":select_time,
            "select_stock":select_stock,
            "hit_rate":hit_rate,
            "value_at_risk":value_at_risk,
            "beta_in":beta_in,
            "bias_in":bias_in,
            "ir":ir,
            "_ry":_sp
        },
        screen= sample_filt
    )

    attach_pipeline(pipe, 'my_pipeline')
    set_max_leverage(1.1)
    schedule_function(
        rebalance,
        date_rule=date_rules.month_start(),
        time_rule=time_rules.market_open()
    )

    # 初始化记录变量
    context.rank_score = pd.Series()
    context.f = pd.Panel()
    context.f_dict = {}
    context.reb_flag = False
    context.B = {}

def model_data(p_data,train_period,rolling,horizon):
    # 错开滞后期
    p_data.iloc[:,:,0] = p_data.iloc[:,:,0].shift(-horizon,axis = 1)
    # 判断是否为rolling,否则是全部训练集作为当期训练集
    if rolling == 1:
        train_data = p_data.iloc[-train_period - 1 -horizon: - horizon,:,:].copy()
    else:
        train_data = p_data.copy()
    # 训练数据集处理
    train_data = train_data.transpose(2,0,1).to_frame(False)
    train_data[np.isinf(train_data)] = np.nan
    train_data = train_data.dropna()
    train_x = train_data.iloc[:, 1:]
    train_y = train_data.iloc[:, 0]
    # 测试数据集处理
    test_x = p_data.iloc[-1,:,1:].copy()
    test_x[np.isinf(test_x)] = np.nan
    test_x = test_x.dropna()
    # 归一化，标准化
    from sklearn.preprocessing import StandardScaler, RobustScaler
    robust_scaler = RobustScaler()
    train_x_pre = robust_scaler.fit_transform(train_x)
    train_x= pd.DataFrame(train_x_pre, columns = train_x.columns, index= train_x.index)
    test_x_pre = robust_scaler.transform(test_x)
    test_x= pd.DataFrame(test_x_pre, columns = test_x.columns, index= test_x.index)
    return train_x,train_y,test_x

def feature_selection(x_train, y_train, x_test):
    from sklearn.feature_selection import SelectFromModel
    clf = ensemble.GradientBoostingRegressor(loss='ls',  learning_rate=0.1)
    clf = clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    X_new_train = model.transform(x_train)
    X_new_test = model.transform(x_test)
    return X_new_train,X_new_test

def lasso_train(x_train, y_train, x_test):
    # Lasso 模型训练 , 使用5折交叉验证， alphas 使用默认集合

    # x_train,x_test = feature_selection(x_train, y_train, x_test)
    lasso_cv = linear_model.LassoCV(alphas=None, cv=5)
    lasso_cv.fit(x_train, y_train)
    B = lasso_cv.coef_
    print B
    print "#"*100
    pred = lasso_cv.predict(x_test)
    return pred,B

def ELM_train(x_train, y_train, x_test):
     # ELM 模型训练 , 使用5折交叉验证， alphas 使用默认集合
    m,n=x_train.shape
    training_matrix=np.append(y_train,x_train.T).reshape(n+1,m).T
    m,n=x_test.shape
    y_test=x_test.iloc[:,0]
    testing_matrix =np.append(y_test,x_test.T).reshape(n+1,m).T
    params = ["sigmoid", 0.01, 15, False]# 激活函数、正则化因子、神经元个数
    elmr = elm.ELMRandom()
#     elmr.search_param(training_matrix, cv="kfold", of="accuracy", eval=5)
#     elmr.print_parameters()
    tr_result = elmr.train(training_matrix)# 训练结果
    te_result = elmr.test(testing_matrix,predicting=True)# 测试集预测结果
    pred= te_result.predicted_targets
    return pred

def knn_train(x_train, y_train, x_test):
    # knn
    # gamma 设为 auto 表示 gamma= 1/feature_num, 代码在fit 函数中
    knn=neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, \
    p=2, metric='minkowski', metric_params=None, n_jobs=1)

    knn.fit(x_train,y_train)
    # 预测结果
    pred=knn.predict(x_test)
    return pred

def adaBoost_train(x_train, y_train, x_test):
    # adaBoost
    base_model = linear_model.LassoCV(alphas=None, cv=5)
    adaBoost=ensemble.AdaBoostRegressor(base_estimator=base_model, n_estimators=1000, learning_rate=1, loss='linear')
    adaBoost.fit(x_train,y_train)
    # 预测结果
    pred=adaBoost.predict(x_test)
    return pred

def bagging_train(x_train, y_train, x_test):
    # bagging
    base_model = linear_model.LassoCV(alphas=None, cv=5)
    bagging=ensemble.BaggingRegressor(base_estimator= base_model, n_estimators=10)
    bagging.fit(x_train,y_train)
    # 预测结果
    pred=bagging.predict(x_test)
    return pred

def gradientBoosting_train(x_train, y_train, x_test):
    # gradientBoosting
    gradientBoosting = ensemble.GradientBoostingRegressor(loss='ls',
                                                         learning_rate=0.1, n_estimators=100, subsample=1.0,\
                                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9,
                                                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    gradientBoosting.fit(x_train,y_train)
    # 预测结果
    pred = gradientBoosting.predict(x_test)
    return pred

# 模型选择字典，代替switch 功能
switcher = {'lasso': lasso_train, 'elm': ELM_train, 'knn': knn_train, 'bagging': bagging_train,
            'adaBoost': adaBoost_train, 'gradientBoosting': gradientBoosting_train}

# 模型选择字典，代替switch 功能
# switcher = {'lasso': lasso_train}
def model_train(x_train, y_train, x_test, model_name):
    return switcher.get(model_name)(x_train, y_train, x_test)

def rebalance(context, data):
    month=context.get_datetime().month
    if month in [2, 5, 8, 11] and context.reb_flag:
        # 原始持仓权重
        weight_old=get_weight(context)
        # 目标持仓权重
        print context.rank_score
        h = np.nanpercentile(context.rank_score, 100 - context.percent,
                           interpolation='higher')
        fund_pool = context.rank_score[context.rank_score >= h]
        # 过滤年轻基金
        longevit = 252 * 1
        fund_data = data.history(fund_pool.index,'close',longevit,'1d')
        selected = fund_pool[fund_data.dropna(how = 'any',axis = 0).columns]
        print selected
        weight_new = pd.Series(0.98/len(selected),index = selected.index)
        # 仓位变动百分比计算
        change = {}
        for stock in weight_new.keys():
            if stock not in weight_old.keys():
                change[stock] = weight_new[stock]
            else:
                change[stock] = weight_new[stock] - weight_old[stock]
        # 下订单
        for stock in sorted(change, key=change.get):
            order_percent(stock, change[stock])
        # 残余头寸清仓处理
        for stock in weight_old.keys():
            if stock not in weight_new.keys():
                order_target_percent(stock,0)
        record(weights = weight_new)
        print '调仓了：'
        print weight_new

def handle_data(context, data):
        context.i  += 1
        record(weights = None)
        record(rebalance_reason = None)
        pipeline_data = pipeline_output('my_pipeline')
        if context.i % 21  == 1 :# 一个月取一次样
            # 从因子中取数，并放入Panel中
            context.f_dict[context.i]= pipeline_data
            context.f = pd.Panel(context.f_dict)
            # print context.f.iloc[:,:,9]
            print str(get_datetime())[:10]
            if context.i > 21 * (context.horizon + context.train_period):# 预测6个月的收益，需要足够6个月才能错位
                context.reb_flag= True
                train_x, train_y, test_x= model_data(context.f.copy(), context.train_period, context.rolling, context.horizon)
                pred = model_train(train_x,train_y, test_x, context.model_name)
                context.rank_score = pd.Series(pred, index= test_x.index)
                context.record_rank[str(get_datetime())[:10]] = context.rank_score
                print context.record_rank
                print   "*"*100

def analyze(context, perf_manual):
    pd.DataFrame(context.record_rank).to_csv(
        './results/fof_ranking_record_hybrid.csv')

