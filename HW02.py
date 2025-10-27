import numpy as np
def f(x):
    return np.exp(x)*(np.sin(x)-2*np.cos(x))
def q(x):
    return 1
#一阶导数边界精度格式
def LEFT_MATRIX_first_order(h=0.1,q=np.exp,lambda1=0,lambda2=0,x=[]):
    m=len(x)-1
    matrix=np.zeros((m+1,m+1))#lenX=m+1
    matrix[0][0]=1+lambda1*h
    matrix[len(x)-1][len(x)-1]=1+lambda2*h

    for i in range(m):
        matrix[i][i+1]=-1
        matrix[i+1][i]=-1

    for i in range(1,m):
        matrix[i][i]=2+h*h*q(x[i])
    #print(matrix)
    return matrix

def RIGHT_VECTOR_first_order(h=0.1,f=np.exp,alpha=0,beta=0,x=[]):
    vector=np.zeros(len(x))
    vector[0]=alpha*h
    vector[len(x)-1]=beta*h
    for i in range(1,len(x)-1):
        vector[i]=h*h*f(x[i])
    #print(vector)
    return vector

def BVP_first_order(h,f=np.exp,alpha=0,beta=0,lambda1=0,lambda2=0,x=[]):
    matrix=LEFT_MATRIX_first_order(h,q,lambda1,lambda2,x=x)
    vector=RIGHT_VECTOR_first_order(h,f,alpha,beta,x=x)
    return np.linalg.solve(matrix,vector)
#二阶精度格式
def LEFT_MATRIX_second_order(h=0.1,q=np.exp,lambda1=0,lambda2=0,x=[]):
    m=len(x)-1
    matrix=np.zeros((m+1,m+1))#lemX=m+1
    matrix[0][0]=1+lambda1*h+h*h*q(x[0])/2
    matrix[m][m]=1+lambda2*h+h*h*q(x[m])/2

    for i in range(m):
        matrix[i][i+1]=-1
        matrix[i+1][i]=-1
    for i in range(1,m):
        matrix[i][i]=2+h*h*q(x[i])
    #print(matrix)
    return matrix

def RIGHT_VECTOR_second_order(h=0.1,f=np.exp,alpha=0,beta=0,x=[]):
    vector=np.zeros(len(x))
    vector[0]=alpha*h+h*h*f(x[0])/2
    vector[len(x)-1]=beta*h+h*h*f(x[len(x)-1])/2
    for i in range(1,len(x)-1):
        vector[i]=h*h*f(x[i])
    #print(vector)
    return vector

def BVP_second_order(h,f=np.exp,alpha=0,beta=0,lambda1=0,lambda2=0,x=[]):
    matrix=LEFT_MATRIX_second_order(h,q,lambda1,lambda2,x=x)
    vector=RIGHT_VECTOR_second_order(h,f,alpha,beta,x=x)
    return np.linalg.solve(matrix,vector)

#example 1.4
if __name__ == "__main__":
    m_list=[160,320,640,1280]
    u=[1.101778,3.341619,6.263717,7.256376]
    for m in m_list:
        h=np.pi/m

        x=[n*h for n in range(m+1)]
        alpha=-1
        beta=-1*(np.exp(np.pi))
        lambda1=0
        lambda2=0
        u=BVP_first_order(h,f,alpha,beta,lambda1,lambda2,x)
        #v=BVP_second_order(h,f,alpha,beta,lambda1,lambda2,x)
        print(u[m//5],u[2*m//5],u[3*m//5],u[4*m//5])
        error=[abs(u[m//5]-1.101778),abs(u[2*m//5]-3.341619),abs(u[3*m//5]-6.263717),abs(u[4*m//5]-7.256376)]
        print(max(error))
        #print(v[m//5],v[2*m//5],v[3*m//5],v[4*m//5])
    n_list=[10,20,40,80,160]
    for n in n_list:
        h=np.pi/n
        x=[k*h for k in range(n+1)]
        alpha=-1
        beta=-1*(np.exp(np.pi))
        lambda1=0
        lambda2=0
        u=BVP_second_order(h,f,alpha,beta,lambda1,lambda2,x)
        print(u[n//5],u[2*n//5],u[3*n//5],u[4*n//5])
        error=[abs(u[n//5]-1.101778),abs(u[2*n//5]-3.341619),abs(u[3*n//5]-6.263717),abs(u[4*n//5]-7.256376)]
        print(max(error))






