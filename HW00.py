import math

def f(x):
    return math.sqrt(2/math.pi)*math.exp(-x*x/2)


def T(f,a,b,n):
    A=[a+i*(b-a)/n for i in range(0,n+1)]
    sum=-(b-a)/n*f(a)*1/2
    for i in A:
        
        sum=sum-f(b)/2*(b-a)/n
    return sum


def S(f,a,b,n):
    A=[a+i*(b-a)/n for i in range(0,n+1)]
    sum=f(a)/3*(b-a)/n
    for i in range(1,len(A)-1):
        if i%2==0:
            sum=sum+4*f(A[i])*(b-a)/(3*n)
        else:
            sum=sum+2*f(A[i])*(b-a)/(3*n)
    sum=sum+f(b)/3*(b-a)/n
    return sum

        
        
def g(x):
    return math.exp(-(1/2)*x*x)

import numpy as np
import math


A=[2,4,8,16,32,64,128,256,512,1024,2048]

def Ti(f,a,b,n):#复化梯形求积
    A=[a+i*(b-a)/n for i in range(0,n+1)]
    sum=-(b-a)/n*f(a)*1/2
    for i in A:
        sum=sum+f(i)*(b-a)/n  
    sum=sum-f(b)/2*(b-a)/n
    return sum

def ERROR_Ti(f,a,b,n):
    return abs(Ti(f,a,b,n)-Ti(f,a,b,10000))

def error_vec(f,a,b,A):
    return [math.log(ERROR_Ti(f,a,b,n)) for n in A]





def T_prime(f,a,b,n,times):
    T=[Ti(f,a,b,n)]
    for i in range(times):
        n=2*n
        T.append(Ti(f,a,b,n))
    return T

def S_second(T):
    S=[]
    for i in range (len(T)-1):
        S.append(T[i]+4/3*(T[i+1]-T[i]))
    return S

def C_third(S):
    C=[]
    for i in range(len(S)-1):
        C.append(S[i]+8/7*(S[i+1]-S[i]))
    return C

def romberg(C):
    R=[]
    for i in range(len(C)-1):
        R.append(C[i]+16/15*(C[i+1]-C[i]))
    return R

def ROMBERG_algo(f,a,b,n,times=4,e=1e-10):
    T=T_prime(f,a,b,n,times)
    S=S_second(T)
    C=C_third(S)
    R=romberg(C)
    if abs(R[1]-R[0])<e:
        return R[1]
    else:
        print(R[0])
        return ROMBERG_algo(f,a,b,2*n,times=4,e=1e-10)
    