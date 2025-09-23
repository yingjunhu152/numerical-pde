import math
import numpy as np
def tridiagonal_for_dirichlet(x,h,q):
    m=len(x)-1
    matrix=np.zeros((m+1, m+1))
    matrix[0][0]=1
    for i in range(1,m):
        matrix[i][i]=2+h*h*q(x[i])
        matrix[i][i-1]=-1
        matrix[i][i+1]=-1
    matrix[m][m]=1
    return matrix

def f(x):
    return math.exp(x)*(math.sin(x)-2*math.cos(x))

def q(x):
    return 1

def mesh(a,b,m):
    x=np.zeros(m+1)
    for i in range(0,m+1):
        x[i]=i*(b-a)/m+a
    return x



def right_vector(h,f,x,alpha,beta):
    m=len(x)-1
    right_vector=np.zeros(m+1)
    right_vector[0]=alpha
    for i in range (1,m):
        right_vector[i]=h*h*f(x[i])
    right_vector[m]=beta
    return right_vector
    
    

def dirichlet_solve(a,b,m,q,f,alpha,beta):
    
    x=mesh(a,b,m)
    left_matrix=tridiagonal_for_dirichlet(x,(b-a)/m,q)
    left_matrix = np.array(left_matrix, dtype=np.float64)
    left_matrix
    right_vectors=right_vector((b-a)/m,f,x,alpha,beta)
    right_vectors = np.array(right_vectors, dtype=np.float64)

    return np.linalg.solve(left_matrix,right_vectors)
    
    



