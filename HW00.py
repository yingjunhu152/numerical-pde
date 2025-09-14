import numpy as np

def cholesky_decomposition(A):
    """
    执行矩阵的Cholesky分解
    
    参数:
    A -- 对称正定矩阵
    
    返回:
    L -- 下三角矩阵，满足 A = L * L^T
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i+1):
            # 计算对角元素
            if i == j:
                s = sum(L[i][k] ** 2 for k in range(i))
                # 检查矩阵是否正定
                if A[i][i] - s < 0:
                    raise ValueError("矩阵不是正定矩阵")
                L[i][i] = np.sqrt(A[i][i] - s)
            # 计算非对角元素
            else:
                s = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - s) / L[j][j]
    
    return L
