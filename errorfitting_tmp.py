import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def load_airfoil_data(file_path):
    """
    加载翼型数据并返回四个独立数组（全部x坐标升序排列）
    输出: 
        x_upper (升序), z_upper, 
        x_lower (升序), z_lower
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    upper_data = []
    lower_data = []
    current_section = None
    
    for line in lines:
        if line.startswith('UpX,UpY'):
            current_section = 'upper'
            continue
        elif line.startswith('LowX,LowY'):
            current_section = 'lower'
            continue
            
        if current_section == 'upper':
            x, y = map(float, line.split(','))
            upper_data.append([x, y])
        elif current_section == 'lower':
            x, y = map(float, line.split(','))
            lower_data.append([x, y])
    
    # 转换为numpy数组
    upper = np.array(upper_data)
    lower = np.array(lower_data)
    
    # 全部按x升序排列
    upper = upper[upper[:, 0].argsort()]  # 上表面按x升序
    lower = lower[lower[:, 0].argsort()]  # 下表面按x升序
    
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]


def CST_parameterization_with_TE_thickness(airfoil_file, N):
    """完全修复的CST参数化函数"""
    # 加载数据
    x_upper, z_upper, x_lower, z_lower = load_airfoil_data(airfoil_file)
    
    # 反转上表面为x降序（后缘→前缘）
    x_upper = x_upper[::-1]
    z_upper = z_upper[::-1]
    
    # 归一化x坐标
    max_x = np.max(x_upper)
    x_norm_upper = x_upper / max_x
    x_norm_lower = x_lower / max_x
    
    # 尾缘厚度
    z_te_upper = z_upper[0]  # 后缘点
    z_te_lower = z_lower[-1]  # 下表面后缘点
    
    # 修正z坐标
    z_modified_upper = z_upper - x_norm_upper * z_te_upper
    z_modified_lower = z_lower - x_norm_lower * z_te_lower
    
    # 类别函数参数
    N1, N2 = 0.5, 1.0
    
    # 类别函数计算
    C_upper = (x_norm_upper)**N1 * (1 - x_norm_upper)**N2
    C_lower = (x_norm_lower)**N1 * (1 - x_norm_lower)**N2
    
    # 构建设计矩阵
    def bernstein_poly(x, N, i):
        return comb(N, i) * (x**i) * ((1-x)**(N-i))
    
    A_upper = np.zeros((len(x_norm_upper), N+1))
    A_lower = np.zeros((len(x_norm_lower), N+1))
    
    for i in range(N+1):
        A_upper[:, i] = bernstein_poly(x_norm_upper, N, i)
        A_lower[:, i] = bernstein_poly(x_norm_lower, N, i)
    
    # 关键修复：正确的矩阵运算方式
    # 原问题：尝试广播 (n_points,N+1) * (N+1,N+1)
    # 新方法：逐元素乘法后求和
    design_matrix_upper = A_upper * C_upper.reshape(-1, 1)  # 形状(n_points,N+1)
    design_matrix_lower = A_lower * C_lower.reshape(-1, 1)
    
    # 添加正则化
    ridge_lambda = 1e-6 * np.eye(N+1)
    
    # 最小二乘拟合
    try:
        CST_coefficients_upper = np.linalg.lstsq(
            design_matrix_upper + ridge_lambda,
            z_modified_upper,
            rcond=None
        )[0]
        
        CST_coefficients_lower = np.linalg.lstsq(
            design_matrix_lower + ridge_lambda,
            z_modified_lower,
            rcond=None
        )[0]
    except np.linalg.LinAlgError as e:
        print(f"阶数 {N} 的最小二乘求解失败: {str(e)}")
        CST_coefficients_upper = np.linalg.pinv(design_matrix_upper) @ z_modified_upper
        CST_coefficients_lower = np.linalg.pinv(design_matrix_lower) @ z_modified_lower
    
    # 计算拟合值
    z_fit_upper = design_matrix_upper @ CST_coefficients_upper + x_norm_upper * z_te_upper
    z_fit_lower = design_matrix_lower @ CST_coefficients_lower + x_norm_lower * z_te_lower
    
    # 计算误差
    def calculate_error(x, z, z_fit):
        weights = np.where(x < 0.2 * max_x, 2.0, 1.0)
        return np.sum(weights * np.abs(z - z_fit))
    
    error_upper = calculate_error(x_upper, z_upper, z_fit_upper)
    error_lower = calculate_error(x_lower, z_lower, z_fit_lower)
    
    return CST_coefficients_upper, CST_coefficients_lower, error_upper, error_lower, error_upper + error_lower

# 调用函数（测试不同阶数）
orders = [6, 8, 10]

for order in orders:
    print(f"\n{'='*50}")
    print(f"测试CST阶数: {order}")
    print(f"{'='*50}")
    
    try:
        # 请替换为您的实际数据文件路径
        CST_coefficients_upper, CST_coefficients_lower, mse_upper, mse_lower, total_error = \
            CST_parameterization_with_TE_thickness("D:\\airfoil_data\\processed_airfoils\\train\\a18.csv", order)  # 假设数据文件为CSV格式
        
        print(f"阶数 {order} 的拟合完成")
        print(f"上表面CST系数数量: {len(CST_coefficients_upper)}")
        print(f"下表面CST系数数量: {len(CST_coefficients_lower)}")
        
    except Exception as e:
        print(f"阶数 {order} 拟合失败: {e}")