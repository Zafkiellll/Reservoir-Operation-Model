

import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import symbols, diff
import numpy as np
import pandas as pd
import sympy
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 13
#采用汛期模型还是非汛期模型的标准在于，用N年一遇的洪量标准来区别。


def diaodu_Defend(W0,Wn,Q1_,Q2_,Qmin_1,Qmax_2,ω,m,μ2,σ2,Rain,n = 2,l = 120,P = 5):
    if Qmin_1 >= Rain:
        Qmin_1 = Qmin_1-Rain
    else:
        Qmin_1 = 0
    print(Qmin_1)
    #定义方程：两个目标各自的偏导数  数量级一定要对的上
    def f1(W1):
        f1 = ω*(m/Wn)*(1-W1/Wn)**(m-1)
        return f1
    def f2(δ):
        f2 = (1-ω)*(1/(σ2*(2*math.pi)**0.5))*np.exp(-δ**2/(2*(σ2)**2))
        return f2
    # 绘制目标函数，直观一点
    # W1_values = np.linspace(0, Wn, 100)
    # δ_values = np.linspace(-5, 5, 100)
    # f1_values = [f1(W1) for W1 in W1_values]
    # f2_values = [f2(δ) for δ in δ_values]
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(W1_values, f1_values, label='f1(W1)')
    # plt.xlabel('W1')
    # plt.ylabel('f1(W1)')
    # plt.title('Function f1(W1)')
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(1, 2, 2)
    # plt.plot(δ_values, f2_values, label='f2(δ)')
    # plt.xlabel('δ')
    # plt.ylabel('f2(δ)')
    # plt.title('Function f2(δ)')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    def qiujie(Δδ=1,h=2,ε = 0.0000001):
        R_0 = n / (l * P)  # 下游可接受风险
        Q1 = Q1_ ##第一阶段来水量 这里第一阶段视为确定性预测，误差为0，因此预测值即为来水
        # Q2_ = Q2_  # 第二阶段总预测来水量
        δ_min = μ2 + σ2 * norm.ppf(1 - R_0)  # 最小防洪安全值，规划设计的防洪风险率对应的防洪安全值， 下游安全过水量与预报来水的差值
        print('最小防洪安全值',δ_min)


        Qmax_2_ = Qmax_2 - δ_min  # 存在预报不确定性时，若下游可接受风险为R_0，则下游允许的最大泄水量
        RFCC = Qmax_2 - Q2_  # 如果是负值的话是不是直接把预蓄水量设成0  这个值
        print('RFCC',RFCC)
        #RFCC是一直变化的
        δ = [δ_min] # 第二阶段水库防洪安全值，自变量2
        W1 = [RFCC - δ[0]] #第一时段预蓄水量，自变量1

        Wmax = Qmax_2_ - Q2_  # 最大预蓄水量

        #下面为牛顿迭代
        j = 0

        if RFCC > δ_min: #当能蓄一部分水的时候
            while True:
                # print(W1[j],f2(δ[j]))
                # print(abs(f1(W1[j]) - f2(δ[j])))
                if j == 0:
                    if abs(f1(W1[j]) - f2(δ[j])) < ε:
                        break

                    else:
                        if f1(W1[j]) < f2(δ[j]):
                            δ.append(δ[j] + Δδ)
                            W1.append(RFCC - δ[j + 1])
                        else:
                            δ.append(δ[j] - Δδ)
                            W1.append(RFCC - δ[j + 1])
                        j += 1
                else:
                    if abs(f1(W1[j])-f2(δ[j])) < ε:
                        break
                    else:
                        if (f1(W1[j])-f2(δ[j]))*(f1(W1[j-1])-f2(δ[j-1])) < 0:
                            Δδ = Δδ/h
                    if f1(W1[j]) < f2(δ[j]):
                        δ.append(δ[j]+Δδ)
                        W1.append(RFCC - δ[j+1])
                    else:
                        δ.append(δ[j]-Δδ)
                        W1.append(RFCC - δ[j+1])
                    j += 1
            W1_star = W1[-1] #W1最优解
            δ_star = δ[-1]

            if δ_star > δ_min and 0 < W1_star < Wmax:
                W1_star = W1_star
                δ_star = δ_star
            else:
                if W1_star < 0:
                    W1_star = 0
                    δ_star = RFCC
                elif W1_star > Wmax:
                    W1_star = Wmax
                    δ_star = RFCC - Wmax
                elif δ_star < δ_min:
                    δ_star = δ_min
                    W1_star = RFCC - δ_min
            D1_star = W0 + Q1 - W1_star
            if D1_star <= Qmin_1:
                if W0 + Q1 - D1_star >= 0:
                    D1_star = Qmin_1 #这里强行等于最小供水量，那么缺的水从哪里补呢 这部分原图存在缺陷，需要修改一下
                    W1_star = W0 + Q1 - D1_star
                    δ_star = RFCC - W1_star


        else:  #当不能保证防洪标准的时候，第一阶段不蓄水
            W1_star = 4.5 #最小库容,这里设置成参数更好
            δ_star = RFCC
            D1_star = Q1



        #上面这俩if存在问题，有可能是顺序放错了，好像也不是顺序放错了。。。。再看看到底咋回事
        print(f"最优解为W1* = {W1_star},D1* = {D1_star} δ* = {δ_star}")
        return W1_star,D1_star,δ_star
    W1_star,D1_star,δ_star = qiujie()



    return W1_star,D1_star,δ_star


from scipy.optimize import minimize
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 13
#采用汛期模型还是非汛期模型的标准在于，用N年一遇的洪量标准来区别。

def diaodu_Benefit(W0,We,Q1_,Q2_,x1_min,x1_max,x2_max,K_max,K_min,μ2,σ2,Rain1,Rain2,Δδ=10,h=2,ε = 0.00001):


    def B1(x): #第一阶段效益函数
        B1 = (x+Rain1)**3 - 3.5*(x+Rain1)**2 + 4*(x+Rain1)  #这里用的是文章中的H效益函数，为高效益函数
        return B1
    def B2(x): #第二阶段效益函数
        B2 = 0.8*(x+Rain2)**3 - 2.8*(x+Rain2)**2 + 3.2*(x+Rain2)  #这里用的是文章中的L效益函数，为低效益函数
        #B2 = (x + Rain) ** 3 - 3.5 * (x + Rain) ** 2 + 4 * (x + Rain)
        return B2

    def B1_(x): # 第一阶段效益函数的一阶偏导
        x1 = symbols('x') # 定义符号变量
        B1_derivative = diff(B1(x1), x1) # 求B1函数的导数
        B1_ = B1_derivative.subs(x1, x) # 求导数在x处的值
        return B1_
    def B2_(x): # 第二阶段效益函数的一阶偏导
        x2 = symbols('x')
        B2_derivative = diff(B2(x2), x2)
        B2_ = B2_derivative.subs(x2, x)
        return B2_

    def B1_3(x): # 第一阶段效益函数的三阶偏导
        x1 = symbols('x')
        B1_derivative = diff(B1(x1), x1, 3)
        B1_ = B1_derivative.subs(x1, x)
        return B1_
    def B2_3(x): # 第二阶段效益函数的三阶偏导
        x2 = symbols('x') # 定义符号变量
        B2_derivative = diff(B2(x2), x2, 3)
        B2_ = B2_derivative.subs(x2, x)
        return B2_
    x1_min = max(x1_min-Rain1,0)

    def objective(x, Q1, Q2, σ):

        B1_ = 2 * 0.01*x[0] ** 2 - 7 * 0.01*x[1] + 4
        B2_ = 3 * 0.8 * 0.01*x[1] ** 2 - 2.8 * 2 * 0.01*x[1] + 3.2
        B2_3 = 0.8 * 3 * 2
        return abs(B1_ - (B2_ + 0.5 * B2_3 * x[1] * σ ** 2))

    # Define the constraints
    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 5},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: -x[1] + 5},
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] - (Q1_ + Q2_)})

    # Perform the optimization

    result = minimize(objective, [0, 0], args=(Q1_, Q2_, σ2), constraints=cons)


    x1_star_star_star = result.x[0]
    x2_star_star_star = result.x[1]
    print(result.x)
    if x1_star_star_star >=0 and x2_star_star_star >= 0:
        x1_star_star = max(x1_star_star_star,x1_min) #若不满足最低供水，则直接等于最低供水

        s1_star_star = W0 + Q1_ - x1_star_star#预设s1值

        s1_star_star = min(s1_star_star,K_max)
        s1_star_star = max(s1_star_star,K_min) #这两步连一块就可以筛选出来s1到底是取边界值还是保留原值了
        s1_star = s1_star_star

        x1_star = max(W0 + Q1_ - s1_star,0) #供不上的时候

        x2_star = Q2_ + s1_star - We
        if x2_star<0:
            x2_star = 0
            #然后从x2的方向倒回去计算s1和x1
            s1_star = x2_star+We-Q2_  #这种情况下肯定是大于0的。因为We已经大于Q2_了 ，水不够会要求s1多蓄点水，那么就有可能超过Kmax了
            s1_star = min(s1_star,K_max)

            x1_star = W0 + Q1_ - s1_star
            #但是如果x1_star倒过来算还是小于最小供水的话...
            x1_star = max(x1_star,x1_min) #蓄不上水，只能先满足一下第一阶段
            s1_star = W0 + Q1_ - x1_star  # 预设s1值

            s1_star = min(s1_star, K_max)
            s1_star = max(s1_star, K_min)  # 这两步连一块就可以筛选出来s1到底是取边界值还是保留原值了
            s1_star = s1_star

            x1_star = max(W0 + Q1_ - s1_star, 0)  # 供不上的时候没法

            x2_star = Q2_ + s1_star - We
    elif x1_star_star_star >= 0 and x2_star_star_star < 0:
        x1_star_star = max(x1_star_star_star, x1_min)  # 若不满足最低供水，则直接等于最低供水

        s1_star_star = W0 + Q1_ - x1_star_star#预设s1值

        s1_star_star = min(s1_star_star,K_max)
        s1_star_star = max(s1_star_star,K_min) #这两步连一块就可以筛选出来s1到底是取边界值还是保留原值了
        s1_star = s1_star_star

        x1_star = max(W0 + Q1_ - s1_star,0)

        x2_star = Q2_ + s1_star - We


    elif x1_star_star_star < 0 and x2_star_star_star >= 0:

        x1_star_star = x1_min  # 若不满足最低供水，则直接等于最低供水


        s1_star_star = W0 + Q1_ - x1_star_star#预设s1值

        s1_star_star = min(s1_star_star,K_max)
        s1_star_star = max(s1_star_star,K_min) #这两步连一块就可以筛选出来s1到底是取边界值还是保留原值了
        s1_star = s1_star_star

        x1_star = max(W0 + Q1_ - s1_star,0)

        x2_star = Q2_ + s1_star - We


    l1_star = W0 + Q1_ - x1_star - s1_star
    print('弃水',l1_star)
    print(x1_star,x2_star,s1_star)
    return x1_star,x2_star,s1_star,l1_star
ω = 0.25  # 兴利效益权重
m = 3  # 曲线系数

μ2 = 0.47
σ2 = 4.586

# 这里应该加个正态分布的误差拟合，导入一下历史资料的预测误差，然后拟合正态分布得到均值和标准差


#第一阶段传递给第二阶段的水的限度为库容上限至末库容的水量
#第二阶段补偿给第一阶段的水的限度为初始库容至库容下限的水量
def SOP(Q1,W0,x1,Wmax,Wmin):
    W = W0+Q1-x1
    W = min(W, Wmax)
    W = max(W, Wmin)  # 这两步连一块就可以筛选出来s1到底是取边界值还是保留原值了
    x1 = max(W0+Q1-W,0)
    W = W0+Q1-x1
    return x1,W

def B1(x):  # 第一阶段效益函数  计算效益
    B1 = (x) ** 3 - 3.5 * (x) ** 2 + 4 * (x)  # 这里用的是文章中的H效益函数，为高效益函数
    return B1

year = [1995,2005,2010,1985,1992,1984]
for y in year:
    # 读取预报来水量
    df = pd.read_excel(f'逐日多时段预测结果{y}.xlsx')
    Q1_ = df.iloc[:, 0].tolist()
    Q1_ = [x * 24*3600/(10**6) for x in Q1_] #数据一般是流量，所以得乘以一天的秒数，单位也得转化
    Q2_ = df.iloc[:, 1:3].sum(axis=1).tolist()
    Q2_ = [x * 24*3600/(10**6) for x in Q2_] ##防洪的第二阶段是两天，兴利的第二阶段只是一天而已，这里恐怕得修改一下，绘或者让非汛期的第二阶段也变成两天
    #读取实际径流数据
    df = pd.read_excel('宝鸡峡-径流.xlsx')
    start_date = f'{y}/1/1 12:00'
    end_date = f'{y}/12/31 12:00'
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    Q1_lastday  = filtered_df['load'].tolist()
    Q1_lastday = [x * 24*3600/(10**6) for x in Q1_lastday] ##读取表格得到过去一日的实际来水量，用于计算实际库容  数据一般是流量，所以得乘以一天的秒数
    #读取预测降雨数据
    Rain = []
    df = pd.read_excel('宝鸡峡-降雨.xlsx')
    start_date = f'{y}/1/1 12:00'
    end_date = f'{y}/12/31 12:00'
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    Rain = filtered_df['load'].tolist()
    Rain_mm = Rain#搞个副本
    Rain = [x / 100 * 1.202 * 10 ** 9 / (10 ** 6) for x in Rain]  ## 实际的降雨补给水量，单位是毫米的话，除以100化为米，然后乘以面积，最后除以单位
    print(Rain)


    from scipy.stats import pearsonr
    corr, p_value = pearsonr(Q1_lastday, Rain)
    print(f'Correlation Coefficient: {corr}, P-value: {p_value}')

    print('相关系数：',corr, p_value)


    x1_min = 0.432 #这个没放到函数里。。。 感觉可以针对不同年份采用不同的最小供水
    x1_max = 4 #第一阶段最大需求水量
    x2_max = 4 #第二阶段最大需求水量
    K_max = 5*10
    K_min = 4.5 #水库最低库容 涉及到一个能否动用死库容的情况，若K_min = 0相当于死库容一点都不能用，考虑考虑吧
    W0 = 3.5*10 #水库初始库容
    We = 5*10 #期望，并不是真实的
    Δδ=10
    h=2
    ε = 0.00001
    W = [W0] #水库库容记录
    D = [] #水库供水记录
    Be = [] #效益记录
    Be_SOP = [] #效益记录
    #满足水量平衡，那就应该没问题
    for i in range(30): #润年366来着
        x1_star=0
        x2_star=0
        W_pre=W[i] + Q1_lastday[i]
        W.append(min(W_pre,K_max))
        D.append(max(W_pre-K_max,0))

    W_SOP=W[:]
    D_SOP=D[:]

    for i in range(30,151): #润年366来着
        x1_star, x2_star, s1_star, l1_star = diaodu_Benefit(W[i], We, Q1_[i], Q2_[i],x1_min, x1_max, x2_max, K_max, K_min, μ2, σ2,Rain[i],Rain[i+1])
        W_pre = W[i] + Q1_lastday[i] - x1_star - l1_star
        W.append(max(min(W_pre,K_max),K_min))
        x1_star = W[i] + Q1_lastday[i]  - l1_star -W[i+1]
        D.append(x1_star)
        Be.append(min(B1(0.01*(x1_star+Rain[i])),1))
        print('水与效益',x1_star+Rain[i],B1(0.01*(x1_star+Rain[i])))



        x1_SOP,W_SOP_pre = SOP(Q1_lastday[i], W_SOP[i], x1_min, K_max, K_min)

        W_SOP.append(W_SOP_pre)
        D_SOP.append(x1_SOP)
        Be_SOP.append(min(B1(0.01*(x1_SOP+Rain[i])),1))
        print('水与效益',x1_SOP+Rain[i],B1(0.01*(x1_SOP+Rain[i])))

    K_max_xun = 4.25*10 #汛期水库最大库容

    Qmax_2 = 10 * 10  # 第二阶段最大下泄水量,这里直接当成水库的下泄流量限制
    # Qmin_1 = 2  # 第一阶段最小需水要求，对下泄的约束，当然，是可能随着时间而变化的，这里如果是定值自然不变

    for i in range(151,304):
        W1_star,D1_star,δ_star = diaodu_Defend(W[i],K_max_xun,Q1_[i],Q2_[i],x1_min,Qmax_2,ω,m,μ2,σ2,Rain[i])

        D1_pre = W[i]+Q1_lastday[i]-W1_star
        if D1_pre<0:
            D1_pre = 0
            W1_star = W[i] + Q1_lastday[i] - D1_pre
        W1_star = min(W1_star,K_max_xun)
        W1_star = max(W1_star,K_min)
        D1_pre = W[i] + Q1_lastday[i] - W1_star
        W.append(W1_star)
        D.append(D1_pre)
        print('来水与泄水',Q1_lastday[i],D[i])


        x1_SOP,W_SOP_pre = SOP(Q1_lastday[i], W_SOP[i], x1_min, K_max_xun, K_min)
        W_SOP.append(W_SOP_pre)
        D_SOP.append(x1_SOP)

    for i in range(304,360):
        x1_star, x2_star, s1_star, l1_star = diaodu_Benefit(W[i], We, Q1_[i], Q2_[i],x1_min, x1_max, x2_max, K_max, K_min, μ2, σ2,Rain[i],Rain[i+1])
        # x1_star, x2_star, s1_star, l1_star = diaodu_Benefit(W[i], We, Q1_[i], Q2_[i], x1_max, x2_max, K_max, K_min, μ2, σ2,Rain[i],Rain[i+1])
        W_pre = W[i] + Q1_lastday[i] - x1_star - l1_star
        W.append(max(min(W_pre,K_max),K_min))
        x1_star = W[i] + Q1_lastday[i]  - l1_star -W[i+1]
        D.append(x1_star)
        Be.append(min(B1(0.01*(x1_star+Rain[i])),1))
        # W_pre = W[i] + Q1_lastday [i] - x1_star - l1_star
        # if K_min < W_pre <= K_max:
        #     W.append(W_pre) #计算实际库容，随后进入下一时段的调度。
        #     D.append(x1_star+l1_star)
        #
        # elif W_pre < K_min:
        #     W.append(0)
        #     D.append(W[i] + Q1_lastday [i])
        #
        # else:
        #     W.append(K_max)
        #     D.append(W[i] + Q1_lastday [i]-K_max)

        x1_SOP,W_SOP_pre = SOP(Q1_lastday[i], W_SOP[i], x1_min, K_max, K_min)
        W_SOP.append(W_SOP_pre)
        D_SOP.append(x1_SOP)
        Be_SOP.append(min(B1(0.01*(x1_SOP+Rain[i])),1))


    df = pd.DataFrame({'list2': Be,'list4': Be_SOP})
    df = pd.DataFrame({'list2': D,'list4': D_SOP})
    # 输出DataFrame到Excel文件
    df.to_excel('ee.xlsx', index=False)

    # 绘制图表
    x1 = range(152)
    x2 = range(151,305)
    x3 = range(304,360)

    #拟合水位库容曲线
    V = [5012.82114, 4969.20707, 4951.76144, 4921.23159, 4890.70174, 4855.81048, 4820.91923, 4799.11219, 4768.58234,
         4738.05249, 4707.52264, 4690.07701, 4655.18575, 4629.01731, 4598.48746, 4572.31902, 4546.15057, 4519.98213,
         4489.45228, 4450.19962, 4424.03117, 4402.22414, 4376.05569, 4336.80303, 4306.27318, 4280.10474, 4249.57489,
         4219.04504, 4192.87659, 4162.34674, 4140.53971, 4105.64845, 4075.1186, 4044.58875, 4018.42031, 3987.89046,
         3957.36061, 3944.27638, 3909.38513, 3874.49387, 3848.32543, 3813.43417, 3787.26573, 3761.09728, 3734.92884,
         3708.7604, 3686.95336, 3652.0621, 3625.89366, 3595.36381, 3573.55677, 3547.38833, 3525.5813, 3486.32863,
         3464.5216, 3433.99175, 3394.73908, 3372.93204, 3342.40219, 3316.23375, 3281.34249, 3259.53546, 3224.6442,
         3202.83716, 3172.30731, 3150.50028, 3119.97043, 3093.80199, 3067.63354, 3037.10369, 3010.93525, 2984.76681,
         2954.23696, 2932.42992, 2897.53866, 2862.64741, 2836.47896, 2810.31052, 2779.78067, 2749.25082, 2718.72097,
         2688.19112, 2666.38408, 2635.85423, 2605.32438, 2583.51735, 2552.9875, 2522.45765, 2491.9278, 2465.75935,
         2435.2295, 2400.33824, 2365.44699, 2339.27854, 2326.19432, 2291.30307, 2278.21884, 2247.68899, 2212.79774,
         2195.35211, 2156.09944, 2125.56959, 2086.31693, 2064.50989, 2033.98004, 2012.17301, 1972.92034, 1942.39049,
         1907.49923, 1876.96938, 1846.43953, 1815.90968, 1789.74124, 1767.9342, 1733.04295, 1706.8745, 1685.06747,
         1663.26043, 1632.73058, 1610.92355, 1584.7551, 1562.94807, 1541.14103, 1501.88837, 1480.08133, 1458.27429,
         1436.46726, 1401.576, 1375.40756, 1349.23912, 1331.79349, 1301.26364, 1283.81801, 1262.01097, 1235.84253,
         1214.03549, 1192.22846, 1166.06001, 1135.53016, 1113.72313, 1078.83187, 1052.66343, 1022.13358, 1000.32654,
         974.1581, 943.62825, 917.4598, 908.73699, 882.56855, 860.76151, 834.59307, 812.78603, 790.979, 769.17196,
         747.36493, 734.2807, 716.83507, 699.38945, 677.58241, 655.77537, 633.96834, 616.52271, 599.07708, 585.99286,
         568.54723, 551.1016, 533.65597, 516.21034, 503.12612, 476.95768
         ]
    h = [636.01621, 635.98094, 635.85749, 635.80458, 635.75167, 635.68113, 635.59295, 635.50477, 635.41659, 635.34605,
         635.27551, 635.16969, 635.11679, 635.02861, 634.9757, 634.88752, 634.81698, 634.7288, 634.62299, 634.57008,
         634.49954, 634.39372, 634.30554, 634.25264, 634.16446, 634.05864, 634.00574, 633.95283, 633.86465, 633.77647,
         633.65302, 633.60011, 633.52957, 633.44139, 633.37085, 633.28267, 633.19449, 633.07104, 633.0005, 632.94759,
         632.87705, 632.78887, 632.70069, 632.63015, 632.54197, 632.47143, 632.34798, 632.29507, 632.22453, 632.13635,
         632.03053, 631.94236, 631.85418, 631.80127, 631.71309, 631.62491, 631.57201, 631.44856, 631.37801, 631.28983,
         631.21929, 631.09584, 631.06057, 630.97239, 630.88421, 630.7784, 630.72549, 630.65495, 630.56677, 630.46096,
         630.39041, 630.30223, 630.19642, 630.09061, 630.0377, 629.94952, 629.86134, 629.75553, 629.68498, 629.57917,
         629.47336, 629.40281, 629.31463, 629.22646, 629.13828, 629.03246, 628.96192, 628.87374, 628.76793, 628.67975,
         628.5563, 628.46812, 628.34467, 628.25649, 628.1154, 628.08013, 627.93905, 627.8685, 627.79796, 627.67451,
         627.58633, 627.44525, 627.35707, 627.25125, 627.11017, 626.98672, 626.8809, 626.73982, 626.66928, 626.59873,
         626.47528, 626.36947, 626.28129, 626.17548, 626.10493, 625.99912, 625.91094, 625.80513, 625.73458, 625.62877,
         625.55823, 625.43478, 625.31133, 625.25842, 625.11733, 625.02915, 624.94097, 624.88807, 624.76462, 624.67644,
         624.53535, 624.44717, 624.32372, 624.23555, 624.14737, 624.04155, 623.93574, 623.81229, 623.68884, 623.56539,
         623.4243, 623.30085, 623.19504, 623.07159, 622.98341, 622.91287, 622.80705, 622.6307, 622.54252, 622.4367,
         622.33089, 622.22507, 622.11926, 621.97817, 621.89, 621.76655, 621.66073, 621.55492, 621.41383, 621.30802,
         621.18457, 621.06112, 620.93767, 620.79658, 620.67313, 620.54968, 620.42623, 620.30278, 620.17933, 620.07352
         ]
    h = h[::-1]
    V = V[::-1]
    from scipy import interpolate
    V_chazhi = interpolate.UnivariateSpline(h, V, s=0)  # 强制通过所有点
    h_chazhi = interpolate.UnivariateSpline(V, h, s=0)
    W_WAN = np.multiply(W, 100)

    H = []
    for i in range(len(W_WAN)):
        if W_WAN[i] == 450:
            H.append(620) #
        else:
            H.append(h_chazhi(W_WAN[i]))



    x = range(360)

    fig = plt.figure(figsize=(12, 5))
    # 调整图边距
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.945, top=0.9)


    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 绘制子图1
    ax1.plot(x1, H[:152], color='blue', alpha=0.5, label='Non-Flood Model')
    ax1.plot(x2, H[151:305], color='red', alpha=0.5, label='Flood Model')
    ax1.plot(x3, H[304:360], color='blue', alpha=0.5)

    H_SOP = []
    W_sop = np.multiply(W_SOP, 100)
    for i in range(len(W_sop)):
        if W_sop[i] == 450:
            H_SOP.append(620)
        else:
            H_SOP.append(h_chazhi(W_sop[i]))
    # H_SOP = h_chazhi(W_sop)
    # H_SOP = [620 if x == 450 else x for x in H_SOP]

    # 创建一个DataFrame
    df = pd.DataFrame({'list1': W_sop, 'list2': H_SOP})

    # 输出DataFrame到Excel文件
    df.to_excel('output.xlsx', index=False)

    ax1.plot(x,H_SOP[:360], color='gray',linestyle='--', label='SOP')
    ax1.set_ylim(620, 638)  # 将刻度增大

    ax1.set_xlabel('T(d)')
    ax1.set_ylabel('Level(m)')
    ax1.set_title(f'Reservoir Water Level Variation Process({y})')
    ax1.legend()

    # 绘制子图2

    ax2.plot(x, Q1_lastday[:360], alpha=0.5,  label='Inflow')
    ax2.plot(x[:360], D[:360], color='red',  alpha=0.5, label='Outflow')
    max_= max(max(D),max(Q1_lastday))
    ax2.set_ylim(0, max_*1.2)  # 将刻度增大

    ax3 = ax2.twinx()  # 与主坐标轴共享 x 轴
    ax3.plot(x, Rain_mm[:360], color='green',  alpha=0.5,label='Rain')
    ax2.set_xlabel('T(d)')
    ax2.set_ylabel('R(Mm³)')
    ax2.set_title(f'Streamflow and Rainfall Process({y})')


    max_rainfall = max(Rain_mm)
    ax3.set_ylim(0, max_rainfall*6)  # 将刻度增大
    ax3.set_ylabel('P(mm)')
    ax3.invert_yaxis()  # 反转 y 轴、
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center left')

    plt.show()

    import pandas as pd

    # 生成时间数据
    time = [f'day{i}' for i in range(1, 361)]  # day1 到 day360
    print(len(H[:360]),len(H_SOP[:360]),len(D[:360]))

    # 创建一个数据字典
    data = {
        'Time': time,
        'Water_level-Model(m)': H[:360],
        'Water_level-SOP(m)': H_SOP[:360],
        'Outflow(Mm³)': D[:360]  # 添加 out 列
    }

    # 使用 pandas 创建 DataFrame
    df = pd.DataFrame(data)

    # 输出到 Excel 文件
    output_file = f'{y}.xlsx'  # 设置输出文件名
    df.to_excel(output_file, index=False)  # index=False 不输出索引列

def createBoundary(self,path,data):
    with open(path,"w",encoding="utf_8") as file_boundary:
        file_boundary.write("Crosssectionwaterlevelflowrelationship\n")
        file_boundary.write("{}\n".format(len(data["zq"])))
        for k in data["zq"]:
            file_boundary.write("{} {}\n".format(k[0],k[1]))
            file_boundary.write(
                "--FitaandbinthecurveZ=a*Q+b\n"
                +" 0.0000.000 0.00039.992"
                )