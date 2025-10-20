import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import numpy as np
import re

def load_airfoil_data(file_path):
    """
    输入： 翼型数据文件路径
    输出：翼型数据数组
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 检测格式类型
    if len(lines) >= 2 and re.match(r'^\d+\.\s+\d+\.$', lines[1]):
        # 20-32C格式（跳过前两行：标题+点数行）
        data_lines = lines[2:]
    else:
        # A18格式或纯数据（跳过可能的注释行）
        data_lines = [line for line in lines if re.match(r'^\s*-?\d+\.\d+', line)]
    
    # 提取有效数据
    data = []
    for line in data_lines:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    
    return np.array(data)

def CST_parameterization_with_TE_thickness(airfoil_file, N):
    
    # 读取翼型数据，跳过标题行
    airfoil_data = load_airfoil_data(airfoil_file)
    x = airfoil_data[:, 0]  # x坐标
    z = airfoil_data[:, 1]  # z坐标（对应公式中的z）

    # 找到前缘点（x坐标最小的点）
    leading_edge_index = np.argmin(x)
    
    # 将数据分为上下表面
    x_upper = x[:leading_edge_index]
    z_upper = z[:leading_edge_index]
    x_lower = x[leading_edge_index:]
    z_lower = z[leading_edge_index:]

    # 归一化x坐标到[0,1]区间
    x_norm_upper = x_upper / np.max(x_upper)
    x_norm_lower = x_lower / np.max(x_lower)

    # 尾缘厚度
    z_te_upper = z_upper[np.argmax(x_upper)]
    z_te_lower = z_lower[np.argmax(x_lower)]

    print('np.argmax(x_upper): ', np.argmax(x_upper))
    print('np.argmax(x_lower): ', np.argmax(x_lower))

    # 从原始z坐标中减去尾缘厚度项 x * z_TE
    # 根据公式 z = C(x)S(x) + x * z_TE，因此 C(x)S(x) = z - x * z_TE
    z_modified_upper = z_upper - x_norm_upper * z_te_upper
    z_modified_lower = z_lower - x_norm_lower * z_te_lower

    # 定义类别函数参数（对于翼型，N1=0.5, N2=1.0）
    N1 = 0.5  # 前缘形状参数
    N2 = 1.0  # 后缘形状参数

    # 构建类别函数 C(x) = x^N1 * (1-x)^N2
    C_upper = (x_norm_upper)**N1 * (1 - x_norm_upper)**N2
    C_lower = (x_norm_lower)**N1 * (1 - x_norm_lower)**N2

    # 构建伯恩斯坦多项式基函数（形状函数的基础）
    A_upper = np.zeros((len(x_norm_upper), N+1))
    A_lower = np.zeros((len(x_norm_lower), N+1))

    for i in range(N+1):
        # 伯恩斯坦多项式项：comb(N,i) * x^i * (1-x)^(N-i)
        A_upper[:, i] = comb(N, i) * (x_norm_upper)**i * (1 - x_norm_upper)**(N-i)
        A_lower[:, i] = comb(N, i) * (x_norm_lower)**i * (1 - x_norm_lower)**(N-i)

    # 使用最小二乘法拟合CST系数
    # 拟合方程：A * C * coefficients = z_modified
    CST_coefficients_upper = np.linalg.lstsq(A_upper * C_upper[:, np.newaxis], 
                                            z_modified_upper, rcond=None)[0]
    CST_coefficients_lower = np.linalg.lstsq(A_lower * C_lower[:, np.newaxis], 
                                            z_modified_lower, rcond=None)[0]

    # 计算拟合值时重新加上尾缘厚度项
    # z_fit = C(x)S(x) + x * z_TE
    z_fit_upper = np.dot(A_upper * C_upper[:, np.newaxis], CST_coefficients_upper) + x_norm_upper * z_te_upper
    z_fit_lower = np.dot(A_lower * C_lower[:, np.newaxis], CST_coefficients_lower) + x_norm_lower * z_te_lower

    # 计算拟合误差（根据图片中的公式7和8）
    def calculate_fitting_error(x_orig, z_orig, z_fit):
        """
        计算拟合误差
        公式7: fitting_error = sum(w * |z_i - z_fit(x_i)|)
        公式8: w = 2 if x_i < 0.2 else 1
        """
        # 计算权重（根据x坐标）
        weights = np.where(x_orig < 0.2, 2.0, 1.0)
        
        # 计算绝对误差并加权
        absolute_errors = np.abs(z_orig - z_fit)
        weighted_errors = weights * absolute_errors
        
        # 总拟合误差
        total_error = np.sum(weighted_errors)
        
        return total_error, weighted_errors

    # 计算上下表面的拟合误差
    fitting_error_upper, weighted_errors_upper = calculate_fitting_error(x_upper, z_upper, z_fit_upper)
    fitting_error_lower, weighted_errors_lower = calculate_fitting_error(x_lower, z_lower, z_fit_lower)
    
    # 总拟合误差
    total_fitting_error = fitting_error_upper + fitting_error_lower

    # 输出误差信息
    print(f"上表面拟合误差: {fitting_error_upper:.6f}")
    print(f"下表面拟合误差: {fitting_error_lower:.6f}")
    print(f"总拟合误差: {total_fitting_error:.6f}")
    print(f"CST阶数 N = {N}")

    return CST_coefficients_upper, CST_coefficients_lower, fitting_error_upper, fitting_error_lower, total_fitting_error

# 调用函数（测试不同阶数）
orders = [6, 8, 10]

for order in orders:
    print(f"\n{'='*50}")
    print(f"测试CST阶数: {order}")
    print(f"{'='*50}")
    
    try:
        CST_coefficients_upper, CST_coefficients_lower, mse_upper, mse_lower, total_error = \
            CST_parameterization_with_TE_thickness('D:\\airfoil_data\\train\\2032c.dat', order)
        
        print(f"阶数 {order} 的拟合完成")
        print(f"上表面CST系数数量: {len(CST_coefficients_upper)}")
        print(f"下表面CST系数数量: {len(CST_coefficients_lower)}")
        
    except Exception as e:
        print(f"阶数 {order} 拟合失败: {e}")
