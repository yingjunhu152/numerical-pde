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
    
import matplotlib.pyplot as plt    
u1=dirichlet_solve(0,math.pi,10,q,f,0,0)
u2=dirichlet_solve(0,math.pi,20,q,f,0,0)
u3=dirichlet_solve(0,math.pi,40,q,f,0,0)
u4=dirichlet_solve(0,math.pi,80,q,f,0,0)
u5=dirichlet_solve(0,math.pi,160,q,f,0,0)


all_y_data = [u1, u2, u3, u4,u5]

labels = ['pi/10', 'pi/20', 'pi/40', 'pi/80','pi/160']

colors = ['blue', 'red', 'green', 'purple','pink'] 

linestyles = ['-', '-', '-', '-','-'] 

# --- 2. 绘制图表 ---
plt.figure(figsize=(10, 6)) # 设置图表尺寸

# 循环绘制每一条折线
for y_data, label, color, linestyle in zip(all_y_data, labels, colors, linestyles):
    
    plt.plot(
        mesh(0,math.pi,len(y_data)-1), 
        y_data, 
        label=label, 
        color=color,        
        linestyle=linestyle, 
        linewidth=2          
    )



plt.xlabel('X-axis Value', fontsize=12)
plt.ylabel('Y-axis Value', fontsize=12)


plt.legend(loc='best', fontsize=10) 

plt.grid(True, linestyle='--', alpha=0.6) 


plt.show()


