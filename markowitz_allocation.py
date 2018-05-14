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