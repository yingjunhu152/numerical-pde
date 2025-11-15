import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# --- 全局定义: 算例 2.1 & 2.2 的问题 ---

def f(x, y):
    """
    源项 f(x, y) = (π² - 1)e^x * sin(πy)
    """
    return (np.pi**2 - 1) * np.exp(x) * np.sin(np.pi * y)

def u_exact(x, y):
    """
    精确解 u(x, y) = e^x * sin(πy)
    """
    return np.exp(x) * np.sin(np.pi * y)

# --- 算例 2.1: 五点差分格式 (O(h^2) 精度) ---
def BVP_FIVE_POINT_GS(m1, m2, tol=2e-10):
    """
    使用五点差分格式和高斯-赛德尔迭代求解。
    """
    h1 = 2.0 / m1
    h2 = 1.0 / m2
    x = np.linspace(0, 2, m1 + 1)
    y = np.linspace(0, 1, m2 + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = np.zeros((m1 + 1, m2 + 1))
    F_source = f(X, Y) 
    
    # 边界条件
    U[0, :] = u_exact(0, Y[0, :]); U[m1, :] = u_exact(2, Y[m1, :])
    U[:, 0] = u_exact(X[:, 0], 0); U[:, m2] = u_exact(X[:, m2], 1)
    
    h1_sq_inv = 1.0 / (h1**2); h2_sq_inv = 1.0 / (h2**2)
    C = 2.0 * (h1_sq_inv + h2_sq_inv)
    error = float('inf')
    U_prev_iter = U.copy()
    
    # Gauss-Seidel 迭代
    while error > tol:
        for i in range(1, m1):
            for j in range(1, m2):
                term_x = h1_sq_inv * (U[i-1, j] + U_prev_iter[i+1, j])
                term_y = h2_sq_inv * (U[i, j-1] + U_prev_iter[i, j+1])
                U[i, j] = (F_source[i, j] + term_x + term_y) / C
        error = np.max(np.abs(U - U_prev_iter))
        U_prev_iter = U.copy()
        
    return U, X, Y
        
# --- 算例 2.2: Richardson 外推 (O(h^4) 精度) ---
def BVP_RICHARDSON_SOLVER(m1, m2, tol=2e-10):
    """
    使用 Richardson 外推法求解 O(h^4) 的解。
    """
    # 求解粗网格 (h) 和细网格 (h/2)
    U_c, X_c, Y_c = BVP_FIVE_POINT_GS(m1, m2, tol)
    U_f, X_f, Y_f = BVP_FIVE_POINT_GS(m1 * 2, m2 * 2, tol)
    U_f_on_coarse = U_f[::2, ::2]
    
    # Richardson 外推公式
    U_richardson = (4.0/3.0) * U_f_on_coarse - (1.0/3.0) * U_c
    return U_richardson, X_c, Y_c

# --- 可视化函数 ---
def plot_solution_and_error(U, U_ex, X, Y, title_prefix, m1, m2):
    """
    绘制数值解和绝对误差的 3D 表面图。
    """
    Error = np.abs(U_ex - U)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"{title_prefix} 解和误差 (m1={m1}, m2={m2})", fontsize=16)

    # Plot 1: 数值解 (3D Surface)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title("数值解 U(x, y)"); ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("U")
    fig.colorbar(surf1, shrink=0.5, aspect=5, ax=ax1)

    # Plot 2: 绝对误差 (3D Surface)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Error, cmap=cm.jet, linewidth=0, antialiased=False)
    ax2.set_title("绝对误差 |U_ex - U|"); ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("Error")
    ax2.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
    fig.colorbar(surf2, shrink=0.5, aspect=5, ax=ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- 主程序: 运行分析和可视化 ---
if __name__ == "__main__":
    
    # --- 运行收敛性测试 (输出表格) ---
    m_pairs_h2 = [(16, 8), (32, 16), (64, 32), (128, 64)]
    
    print("--- 运行 算例 2.1 (五点差分, O(h^2)) ---")
    last_error_h2 = -1
    print(f"{'(m1, m2)':^12} | {'Max Error (E_inf)':^18} | {'Ratio (E_2h / E_h)':^18}")
    print("-" * 52)
    for m1, m2 in m_pairs_h2:
        U, X, Y = BVP_FIVE_POINT_GS(m1, m2)
        U_ex = u_exact(X, Y)
        error = np.max(np.abs(U_ex - U))
        ratio_str = "N/A" if last_error_h2 <= 0 else f"{last_error_h2 / error:.4f}"
        print(f"({m1:<3}, {m2:<3}) | {error:^18.6e} | {ratio_str:^18}")
        last_error_h2 = error

    print("\n--- 运行 算例 2.2 (Richardson, O(h^4)) ---")
    m_pairs_h4 = [(16, 8), (32, 16), (64, 32)]
    last_error_h4 = -1
    print(f"{'(m1, m2)':^12} | {'Max Error (E_tilde)':^18} | {'Ratio (E_2h / E_h)':^18}")
    print("-" * 52)
    for m1, m2 in m_pairs_h4:
        U_tilde, X_c, Y_c = BVP_RICHARDSON_SOLVER(m1, m2) 
        U_ex_c = u_exact(X_c, Y_c)
        error = np.max(np.abs(U_ex_c - U_tilde))
        ratio_str = "N/A" if last_error_h4 <= 0 else f"{last_error_h4 / error:.4f}"
        print(f"({m1:<3}, {m2:<3}) | {error:^18.6e} | {ratio_str:^18}")
        last_error_h4 = error

    # --- 可视化：选择 h=1/32 的 Richardson 外推解 (m1=64, m2=32) ---
    m_plot = 64
    n_plot = 32
    U_h4, X_h4, Y_h4 = BVP_RICHARDSON_SOLVER(m_plot, n_plot)
    U_ex_h4 = u_exact(X_h4, Y_h4)
    print(f"\n--- 正在生成 Richardson 外推解的可视化 (m1={m_plot}, m2={n_plot}) ---")
    plot_solution_and_error(U_h4, U_ex_h4, X_h4, Y_h4, "Richardson 外推 (O(h⁴))", m_plot, n_plot)
    print("可视化代码已生成，请在支持 Matplotlib 的环境中运行查看 3D 表面图。")