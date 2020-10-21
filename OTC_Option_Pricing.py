
#场外期权定价
import numpy as np
import pandas as pd 
from scipy.stats import norm


#%%
#生成蒙特卡洛
def St_MC(underlying_price_0, risk_free_rate, T, sigma):
    M = 100000
    n = 252   
    dt = T / n
    #np.random.seed(1)
    S = np.zeros((n+1, M))
    S[0] = underlying_price_0  
    
    for t in range(1, n+1):       
        z = np.random.standard_normal(M)
        S[t] = S[t-1] * np.exp((risk_free_rate - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
          
    return S.T

#%%
#二元期权定价模型
def Binary_Option(underlying_price, strike_price, risk_free_rate, sigma, term, option_type):  
    d1=(np.log(underlying_price/strike_price)+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d2=d1-sigma*np.sqrt(term)
    if option_type == 1:
        price = np.exp(-risk_free_rate*term)*norm.cdf(d2)
    else:
        price = np.exp(-risk_free_rate*term)*norm.cdf(-d2)  
    return price
        

#%%
#牛熊市价差期权定价
def bscall(underlying_price, strike_price, risk_free_rate, sigma, term):    
    d1=(np.log(underlying_price/strike_price)+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d2=d1-sigma*np.sqrt(term)
    call = underlying_price*norm.cdf(d1)-strike_price*np.exp(-risk_free_rate*term)*norm.cdf(d2)
    return call


def bsput(underlying_price, strike_price, risk_free_rate, sigma, term):  
    d1=(np.log(underlying_price/strike_price)+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d2=d1-sigma*np.sqrt(term)
    put = -underlying_price*norm.cdf(-d1)+strike_price*np.exp(-risk_free_rate*term)*norm.cdf(-d2)  
    return put


def Bull_Spread(underlying_price, strike_price_1, strike_price_2, risk_free_rate, sigma, term, option_type):  
    if option_type == 1:
        # K1<K2
        p1 = bscall(underlying_price, strike_price_1, risk_free_rate, sigma, term)
        p2 = bscall(underlying_price, strike_price_2, risk_free_rate, sigma, term)
        price = p1 - p2
    else:
        # K1<K2
        p1 = bsput(underlying_price, strike_price_1, risk_free_rate, sigma, term)
        p2 = bsput(underlying_price, strike_price_2, risk_free_rate, sigma, term)
        price = p1 - p2
    return price


def Bear_Spread(underlying_price, strike_price_1, strike_price_2, risk_free_rate, sigma, term, option_type):  
    if option_type == 1:
        # K1<K2
        p1 = bscall(underlying_price, strike_price_1, risk_free_rate, sigma, term)
        p2 = bscall(underlying_price, strike_price_2, risk_free_rate, sigma, term)
        price = p2 - p1
    else:
        # K1<K2
        p1 = bsput(underlying_price, strike_price_1, risk_free_rate, sigma, term)
        p2 = bsput(underlying_price, strike_price_2, risk_free_rate, sigma, term)
        price = p2 - p1
    return price



#%%
#单边障碍期权定价
def N(x):
    return norm.cdf(x)

def Cal_Prob(underlying_price,K,r,vol,T,B,option_type):
    
    M = 100000
    n = 100  
    dt = T / n    
    S = np.zeros((n+1, M))
    S[0] = underlying_price  
    
    for t in range(1, n+1):    
        z = np.random.standard_normal(M)    
        S[t] = S[t-1] * np.exp((r - 0.5*vol**2) * dt + vol * np.sqrt(dt) * z)         
    St = S.T
    
    num_up_and_out = 0
    num_up_and_in = 0
    num_down_and_out = 0
    num_down_and_in = 0
    for i in range(M):    
        if (option_type == 'up_and_out_call') | (option_type == 'up_and_out_put'):
            if St[i,:].max() > B:
                num_up_and_out += 1
        elif (option_type == 'up_and_in_call') | (option_type == 'up_and_in_put'):
            if St[i,:].max() < B:
                num_up_and_in += 1
        elif (option_type == 'down_and_out_call') | (option_type == 'down_and_out_put'):
            if St[i,:].min() < B:
                num_down_and_out += 1         
        elif (option_type == 'down_and_in_call') | (option_type == 'down_and_in_put'):
            if St[i,:].min() > B:
                num_down_and_in += 1          
        else:
            break
                
    prob_up_and_out = num_up_and_out/M
    prob_up_and_in = num_up_and_in/M
    prob_down_and_out = num_down_and_out/M
    prob_down_and_in = num_down_and_in/M
         
    return prob_up_and_out, prob_up_and_in, prob_down_and_out, prob_down_and_in

def Barrier_Option_Pricing(underlying_price, strike_price, risk_free_rate, sigma, term, barrier, fixed_return, option_type)   : 
    
    a = (barrier/underlying_price)**(-1+2*risk_free_rate/sigma**2)
    b = (barrier/underlying_price)**(1+2*risk_free_rate/sigma**2)
    
    if fixed_return == 0:        
        rebate_up_and_out = 0
        rebate_up_and_in = 0
        rebate_down_and_out = 0
        rebate_down_and_in = 0
    else:    
        prob = Cal_Prob(underlying_price,strike_price,risk_free_rate,sigma,term,barrier,option_type)
        rebate_up_and_out = np.exp(-risk_free_rate*term)*fixed_return*prob[0]
        rebate_up_and_in = np.exp(-risk_free_rate*term)*fixed_return*prob[1]
        rebate_down_and_out = np.exp(-risk_free_rate*term)*fixed_return*prob[2]
        rebate_down_and_in = np.exp(-risk_free_rate*term)*fixed_return*prob[3]
 
    d1=(np.log(underlying_price/strike_price)+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d2=d1-sigma*np.sqrt(term)
    d3=(np.log(underlying_price/barrier)+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d4=(np.log(underlying_price/barrier)+(risk_free_rate-0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d5=(np.log(underlying_price/barrier)-(risk_free_rate-0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d6=(np.log(underlying_price/strike_price)-(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d7=(np.log((underlying_price*strike_price)/barrier**2)-(risk_free_rate-0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    d8=(np.log((underlying_price*strike_price)/barrier**2)-(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
    
    if option_type == 'up_and_out_call':
        if strike_price>barrier:
            price = 0
        else:
            price0 = underlying_price*(N(d1)-N(d3)-b*(N(d6)-N(d8)))-strike_price*np.exp(-risk_free_rate*term)*(N(d2)-N(d4)-a*(N(d5)-N(d7)))
            price = price0 + rebate_up_and_out
                  
    elif option_type == 'up_and_in_call':
        if strike_price>barrier:
            price = 0
        else:
            price0 = underlying_price*(N(d3)+b*(N(d6)-N(d8)))-strike_price*np.exp(-risk_free_rate*term)*(N(d4)+a*(N(d5)-N(d7)))
            price = price0 + rebate_up_and_in

    elif option_type == 'down_and_out_put':
        if strike_price>barrier:
            price0 = -underlying_price*(N(d3)-N(d1)-b*(N(d8)-N(d6)))+strike_price*np.exp(-risk_free_rate*term)*(N(d4)-N(d2)-a*(N(d7)-N(d5)))
            price = price0 + rebate_down_and_out
        else:
            price = 0
            
    elif option_type == 'down_and_in_put':
        if strike_price>barrier:
            price0 = -underlying_price*(1-N(d3)+b*(N(d8)-N(d6)))+strike_price*np.exp(-risk_free_rate*term)*(1-N(d4)+a*(N(d7)-N(d5)))
            price = price0 + rebate_down_and_in
        else:
            price = 0        
         
    elif option_type == 'down_and_out_call':
        if strike_price>barrier:
            price0 = underlying_price*(N(d1)-b*(1-N(d8)))-strike_price*np.exp(-risk_free_rate*term)*(N(d2)-a*(1-N(d7)))
            price = price0 + rebate_down_and_out
        else:
            price0 = underlying_price*(N(d3)-b*(1-N(d6)))-strike_price*np.exp(-risk_free_rate*term)*(N(d4)-a*(1-N(d5)))
            price = price0 + rebate_down_and_out

    elif option_type == 'down_and_in_call':
        if strike_price>barrier:
            price0 = underlying_price*(b*(1-N(d8)))-strike_price*np.exp(-risk_free_rate*term)*a*(1-N(d7))
            price = price0 + rebate_down_and_in
        else:
            price0 = underlying_price*(N(d1)-N(d3)+b*(1-N(d6)))-strike_price*np.exp(-risk_free_rate*term)*(N(d2)-N(d4)+a*(1-N(d5)))
            price = price0 + rebate_down_and_in

    elif option_type == 'up_and_out_put':
        if strike_price>barrier:
            price0 = -underlying_price*(1-N(d3)+b*N(d6))+strike_price*np.exp(-risk_free_rate*term)*(1-N(d4)-a*N(d5)) 
            price = price0 + rebate_up_and_out
        else:
            price0 = -underlying_price*(1-N(d1)-b*N(d8))+strike_price*np.exp(-risk_free_rate*term)*(1-N(d2)-a*N(d7)) 
            price = price0 + rebate_up_and_out

    elif option_type == 'up_and_in_put':
        if strike_price>barrier:
            price0 = -underlying_price*(N(d3)-N(d1)+b*N(d6))+strike_price*np.exp(-risk_free_rate*term)*(N(d4)-N(d2)+a*N(d5))
            price = price0 + rebate_up_and_in
        else:
            price0 = -underlying_price*b*N(d8)+strike_price*np.exp(-risk_free_rate*term)*a*N(d7)
            price = price0 + rebate_up_and_in
            
    return price



#%%
#亚式期权定价模型 离散几何平均定价
def Asian_Geometric_Mean(underlying_price, strike_price, risk_free_rate, sigma, term, option_type):
    sigma_g = np.sqrt((1/3)*sigma**2)
    mu_g = 1/2 *(risk_free_rate-0.5*sigma**2)+0.5*sigma_g**2
    d1 = (np.log(underlying_price/strike_price) + (mu_g+0.5*sigma_g**2)*term)/(sigma_g*np.sqrt(term))
    d2 = d1-sigma_g*np.sqrt(term)
    if option_type == 1:
        price = underlying_price*np.exp((mu_g-risk_free_rate)*term)*norm.cdf(d1)-strike_price*np.exp(-risk_free_rate*term)*norm.cdf(d2)
    else:
        price = strike_price*np.exp(-risk_free_rate*term)*norm.cdf(-d2)-underlying_price*np.exp((mu_g-risk_free_rate)*term)*norm.cdf(-d1)
    return price


#%%
#亚式期权定价模型 离散算术平均定价
def Asian_Arithmetic_Mean(underlying_price, strike_price, risk_free_rate, sigma, term, option_type): 
    St = St_MC(underlying_price, risk_free_rate, term, sigma)
    M = St.shape[0]
    v = np.zeros(M)# 初始化每次模拟的期权价格向量
    
    for i in range(M):
        # 分成call和put正常定价
        if option_type == 1:
            v[i] = max((St[i,:].mean()-strike_price),0)
        else:
            v[i] = max((strike_price - St[i,:].mean()),0)            
            
    price = np.exp(-risk_free_rate*term) * v.mean()
    return price


#%%
#双鲨期权定价
def Double_Barrier_Pricing(underlying_price, strike_price_1, strike_price_2, risk_free_rate, term, sigma, barrier_1, barrier_2, fixed_return):
    # B1 < K1 < K2 < B2
    St = St_MC(underlying_price, risk_free_rate, term, sigma)
    M = St.shape[0]
    v = np.zeros(M)    
    
    for i in range(M):
        
        if (St[i,:].min() < barrier_1) | (St[i,:].max() > barrier_2):
            v[i] = fixed_return
            
        elif (St[i,:].min() >= barrier_1) & (St[i,:].max() <= barrier_2):
            if St[i,-1] <= strike_price_1:
                v[i] = strike_price_1 - St[i,-1]
            elif (St[i,-1] > strike_price_1) & (St[i,-1] < strike_price_2):
                v[i] = fixed_return
            elif St[i,-1] >= strike_price_2:
                v[i] = St[i,-1] - strike_price_2
            else:
                print('error')
        else:
            print('error')
    price = np.exp(-risk_free_rate * term) * v.mean()        
    return price


#%%
#雪球期权定价
    
#生成蒙特卡洛
def St_MC_SNOW(underlying_price, r, T, sigma):
    M = 10000
    n = 252   
    dt = T / n
    #np.random.seed(1)
    S = np.zeros((n+1, M))
    S[0] = underlying_price  
    
    for t in range(1, n+1):       
        z = np.random.standard_normal(M)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
          
    return S.T

def Snowball(underlying_price, risk_free_rate, term, sigma, knockin_rate, knockout_rate, fixed_return, freq):
    x = []
    St = St_MC_SNOW(underlying_price, risk_free_rate, term, sigma)
    col_no = St.shape[1]
    jump = col_no // freq
    i = 1
    while jump * i <= col_no:
        x.append(jump * i )
        i += 1
    St = pd.DataFrame(St)    
    St_m = St.iloc[:][x]
    M = St.shape[0]
    v = np.zeros(M)
    c = St.shape[1]
    index_list = []
    z = []
    
    for i in range(M):
        index = [i for i,x in enumerate(St_m.iloc[i,:]) if x >= (underlying_price * knockout_rate)]
        if len(index) == 0:
            index0 = -1
        else:
            index0 = index[0]
        index_list.append(index0)
        index_list1 = [(index_no+1) * jump for index_no in index_list] 
        z = [indi /252 * fixed_return * np.exp(-risk_free_rate * indi /252 * term)for indi in index_list1]
        if (St_m.iloc[i,:].max()< (underlying_price * knockout_rate)) & (St.iloc[i,:].min()> (underlying_price * knockin_rate)):
            v[i] = fixed_return * np.exp(-risk_free_rate * term)
        elif (St_m.iloc[i,:].max()< (underlying_price * knockout_rate)) & (St.iloc[i,:].min()<= (underlying_price * knockin_rate)):
            if St.iloc[i,c-1] >= underlying_price:
                v[i] = 0
            elif St.iloc[i,c-1] < underlying_price:
                v[i] = (St.iloc[i,c-1] - underlying_price) /underlying_price  * np.exp(-risk_free_rate * term)
            else:
                print('Error')
        else:
            v[i] = z[i]
    price = v.mean()
    return price



#%%
#one_touch 二元期权定价

def One_Touch_Binary(underlying_price, risk_free_rate, sigma, term, barrier, fixed_return, option_type):
    St = St_MC(underlying_price,risk_free_rate,term,sigma)
    M = St.shape[0]
    v = np.zeros(M) # 初始化每次模拟的期权价格向量
    if option_type == 1:
        for i in range(M):
            if St[i,:].max() >= barrier: 
                v[i] = fixed_return            
            else: # not touch down
                v[i] = 0
    if option_type == -1:
        for i in range(M):
            if St[i,:].min() <= barrier:
                v[i] = fixed_return            
            else: # not touch down
                v[i] = 0
    price = np.exp(-risk_free_rate * term) * v.mean()
    
    return price

#%%
#百慕大期权定价
    
def Bermuda_Option(underlying_price, strike_price, risk_free_rate, sigma, term, option_type):
    if option_type == 1:
        c1 = bscall(underlying_price, strike_price, risk_free_rate, sigma, term)
        d1=(np.log(underlying_price/(strike_price + c1))+(risk_free_rate + 0.5*sigma**2)* term)/(sigma*np.sqrt(term))
        d2=d1-sigma*np.sqrt(term)
        price = c1*np.exp(-risk_free_rate*term) + underlying_price * norm.cdf(d1) - (strike_price + c1)*np.exp(-risk_free_rate*term) * norm.cdf(d2)
    else:
        p1 = bsput(underlying_price, strike_price, risk_free_rate, sigma, term)
        f1=(np.log(underlying_price/(strike_price - p1))+(risk_free_rate+0.5*sigma**2)*term)/(sigma*np.sqrt(term))
        f2=f1-sigma*np.sqrt(term)
        price = p1*np.exp(-risk_free_rate*term) - underlying_price * norm.cdf(-f1) + (strike_price - p1)*np.exp(-risk_free_rate*term) * norm.cdf(-f2)
    return price



#%%
#凤凰期权定价
    
#生成蒙特卡洛
def St_MC_phoenix(underlying_price, r, T, sigma):
    M = 10000
    n = 252   
    dt = T / n
    #np.random.seed(1)
    S = np.zeros((n+1, M))
    S[0] = underlying_price  
    
    for t in range(1, n+1):       
        z = np.random.standard_normal(M)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
          
    return S.T

def Phoenix(underlying_price, risk_free_rate, term, sigma, knockin_rate, knockout_rate, fixed_return, freq):
    x = []
    St = St_MC_phoenix(underlying_price, risk_free_rate, term, sigma)
    col_no = St.shape[1]
    jump = col_no // freq
    i = 1
    while jump * i <= col_no:
        x.append(jump * i )
        i += 1
    St = pd.DataFrame(St)    
    St_m = St.iloc[:][x]
    M = St.shape[0]
    v = np.zeros(M)
    
    for i in range(M):
        index_out = [i for i,x in enumerate(St_m.iloc[i,:]) if x > (underlying_price * knockout_rate)]
        index_in = [i for i,x in enumerate(St_m.iloc[i,:]) if x < (underlying_price * knockin_rate)]
        if len(index_out) != 0 & len(index_in) == 0:
            v[i] = (index_out[0] + 1) * fixed_return/12
        elif len(index_out) != 0 & len(index_in) != 0:
            if index_out[0] > index_in[0]:
                month_eft = len([x for x in index_in if x < index_out[0]])
                v[i] = (index_out[0] - month_eft + 1) * fixed_return/12
            else:
                v[i] = (index_out[0] + 1) * fixed_return/12
        elif len(index_out) == 0 & len(index_in) == 0:
            v[i] = fixed_return
        elif len(index_out) == 0 & len(index_in) != 0:
            v[i] = (St_m.iloc[i,-1] - underlying_price)/underlying_price + (12 - len(index_in)) * fixed_return/12
        else:
            print('Error')
    price = underlying_price * np.exp(-risk_free_rate * term) * v.mean()
    return price
         
   
#%%
#区间累计期权定价
    
#生成蒙特卡洛
def St_MC_Accumulator(underlying_price, r, T, sigma):
    M = 100000
    n = 12   
    dt = T / n
    #np.random.seed(1)
    S = np.zeros((n+1, M))
    S[0] = underlying_price  
    
    for t in range(1, n+1):       
        z = np.random.standard_normal(M)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
          
    return S.T


def Accumulator_Option(underlying_price, risk_free_rate, term, sigma, strike_price, knock_out_price, amount):
    St = St_MC_Accumulator(underlying_price, risk_free_rate, term, sigma)
    St = pd.DataFrame(St)    
    M = St.shape[0]
    payoff = 0 
    x = []
    
    for i in range(M):
        price = St.iloc[i,1:]
        for j in range(1,len(price)+1):
            if price[j] > knock_out_price:
                break
            if strike_price <= price[j] <= knock_out_price:
                payoff += amount * (price[j] - strike_price) *np.exp(-risk_free_rate *j *1/term)
            else:
                payoff += amount * (price[j] - strike_price) * 2 *np.exp(-risk_free_rate *j *1/term)
        x.append(payoff)
        payoff = 0 
    
    price = np.mean(x)
    return price



#%%
#气囊期权定价
    
#生成蒙特卡洛
def St_MC_Airbag(underlying_price, r, T, sigma):
    M = 10000
    n = 252   
    dt = T / n
    #np.random.seed(1)
    S = np.zeros((n+1, M))
    S[0] = underlying_price  
    
    for t in range(1, n+1):       
        z = np.random.standard_normal(M)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2) * dt + sigma * np.sqrt(dt) * z)
          
    return S.T


def Airbag(underlying_price, risk_free_rate, term, sigma, knockin_rate, no_knockin_enter_rate, knockin_enter_rate, freq):
    x = []
    St = St_MC_Airbag(underlying_price, risk_free_rate, term, sigma)
    col_no = St.shape[1]
    jump = col_no // freq
    i = 1
    while jump * i <= col_no:
        x.append(jump * i )
        i += 1
    St = pd.DataFrame(St)    
    St_m = St.iloc[:][x]
    M = St.shape[0]
    v = np.zeros(M)
    
    for i in range(M):
        index = [i for i,x in enumerate(St_m.iloc[i,:]) if x < (underlying_price * knockin_rate)]
        if len(index) != 0:
            v[i] = St_m.iloc[i,-1] / underlying_price * knockin_enter_rate
        else:
            if St_m.iloc[i,-1] < underlying_price:
                v[i] = 0
            else:
                v[i] = (St_m.iloc[i,-1] / underlying_price) * no_knockin_enter_rate
    price = np.exp(-risk_free_rate * term) * v.mean()
    return price




#%%
if __name__ == '__main__':          
    binary_price = Binary_Option(15,10,0.04,0.2,1,1)
    bull_price = Bull_Spread(15,10,12,0.04,0.2,1,1)
    bear_price = Bear_Spread(15,10,12,0.04,0.2,1,-1)
    down_and_out_call_price = Barrier_Option_Pricing(100, 100, 0.1, 0.3, 0.2, 85, 0, 'down_and_out_call')
    geo_mean = Asian_Geometric_Mean(10,10,0.04,0.2,1,1)
    ari_mean = Asian_Arithmetic_Mean(10,10,0.04,0.2,1,1)
    double_barrier_price = Double_Barrier_Pricing(100,100,105,0.1,0.2,0.3,85,115,5)
    snow_price = Snowball(10, 0.04, 1, 0.2, 0.9, 1.2, 0.21, 12)
    one_touch_binary_price = One_Touch_Binary(100,0.0289,0.2706,0.2466,110,0.05*5000,1)
    bermuda_option_price = Bermuda_Option(100,80,0.04,0.6,0.5,-1)
    phoenix_price = Phoenix(10, 0.04, 1, 0.2, 0.9, 1.2, 0.21, 12)
    accumulator_option_price = Accumulator_Option(100,0.04,12,0.05,90,105,1000)
    airbag_price = Airbag(1000, 0.04, 0.5, 0.2, 0.8, 0.78, 1, 12)

    