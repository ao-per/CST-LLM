import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 定义函数（x≥0时，sign(x)=1，|x|=x，可简化表达式）
def f(x):
    return 0.5 * x + 0.5 * x **(1/3)  # x∈[0,1]时，sign(x)=1，|x|=x，简化公式

# 生成x数据（0到1，取1000个点确保平滑）
x = np.linspace(0, 1, 1000)

# 计算y值
y = f(x)

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制函数曲线
plt.plot(x, y, color='blue', linewidth=2, label=r'$f(x) = 0.5x + 0.5x^{1/3} \quad (0 \leq x \leq 1)$')

# 添加坐标轴辅助线
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 添加网格
plt.grid(True, alpha=0.3)

# 添加标题和标签
plt.title('函数在x∈[0,1]区间的图像', fontsize=12)
plt.xlabel('x', fontsize=10)
plt.ylabel('f(x)', fontsize=10)

# 添加图例
plt.legend()

# 调整布局并显示
plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import comb
# import matplotlib

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# def load_airfoil_data(file_path):
#     """
#     输入： 翼型数据文件路径（包含UpX,UpY和LowX,LowY分隔的上下表面数据）
#     输出：翼型数据数组（按x坐标从后缘到前缘再到后缘的顺序排列）
#     """
#     with open(file_path, 'r') as f:
#         lines = [line.strip() for line in f if line.strip()]
    
#     # 分离上下表面数据
#     upper_data = []
#     lower_data = []
#     current_section = None
    
#     for line in lines:
#         # 识别数据段标识
#         if line.startswith('UpX,UpY'):
#             current_section = 'upper'
#             continue
#         elif line.startswith('LowX,LowY'):
#             current_section = 'lower'
#             continue
            
#         # 处理数据行
#         if current_section is not None:
#             try:
#                 x, y = map(float, line.split(','))
#                 if current_section == 'upper':
#                     upper_data.append([x, y])
#                 else:
#                     lower_data.append([x, y])
#             except ValueError:
#                 continue  # 跳过格式错误的行
    
#     # 转换为numpy数组
#     upper = np.array(upper_data)
#     lower = np.array(lower_data)
    
#     # 确保上下表面数据按x坐标排序（从后缘到前缘）
#     # 找到前缘点（x最小点）的索引
#     if len(upper) > 0 and len(lower) > 0:
#         # 合并数据：上表面（从后缘到前缘）+ 下表面（从前缘到后缘，排除重复的前缘点）
#         leading_edge_idx_upper = np.argmin(upper[:, 0])
#         leading_edge_idx_lower = np.argmin(lower[:, 0])
        
#         # 按顺序拼接：上表面后缘到前缘 + 下表面前缘到后缘（跳过重复的前缘点）
#         airfoil_data = np.vstack([
#             upper[:leading_edge_idx_upper + 1],  # 上表面从后缘到前缘
#             lower[leading_edge_idx_lower + 1:]   # 下表面从前缘到后缘（排除前缘点）
#         ])
#         return airfoil_data
#     else:
#         raise ValueError("文件中未找到有效的上下表面数据")

# def CST_parameterization_with_TE_thickness(airfoil_file, N):
    
#     # 读取翼型数据
#     airfoil_data = load_airfoil_data(airfoil_file)
#     x = airfoil_data[:, 0]  # x坐标
#     z = airfoil_data[:, 1]  # z坐标（对应公式中的z）

#     # 找到前缘点（x坐标最小的点）
#     leading_edge_index = np.argmin(x)
    
#     # 将数据分为上下表面
#     x_upper = x[:leading_edge_index + 1]  # 包含前缘点
#     z_upper = z[:leading_edge_index + 1]
#     x_lower = x[leading_edge_index:]      # 包含前缘点
#     z_lower = z[leading_edge_index:]

#     # 归一化x坐标到[0,1]区间
#     max_x = np.max(x)  # 后缘x坐标应为1.0左右
#     x_norm_upper = x_upper / max_x
#     x_norm_lower = x_lower / max_x

#     # 尾缘厚度（x最大处的z值）
#     z_te_upper = z_upper[np.argmax(x_upper)]
#     z_te_lower = z_lower[np.argmax(x_lower)]

#     # 从原始z坐标中减去尾缘厚度项 x * z_TE
#     # 根据公式 z = C(x)S(x) + x * z_TE，因此 C(x)S(x) = z - x * z_TE
#     z_modified_upper = z_upper - x_norm_upper * z_te_upper
#     z_modified_lower = z_lower - x_norm_lower * z_te_lower

#     # 定义类别函数参数（对于翼型，N1=0.5, N2=1.0）
#     N1 = 0.5  # 前缘形状参数
#     N2 = 1.0  # 后缘形状参数

#     # 构建类别函数 C(x) = x^N1 * (1-x)^N2
#     C_upper = (x_norm_upper)**N1 * (1 - x_norm_upper)**N2
#     C_lower = (x_norm_lower)**N1 * (1 - x_norm_lower)**N2

#     # 构建伯恩斯坦多项式基函数（形状函数的基础）
#     A_upper = np.zeros((len(x_norm_upper), N+1))
#     A_lower = np.zeros((len(x_norm_lower), N+1))

#     for i in range(N+1):
#         # 伯恩斯坦多项式项：comb(N,i) * x^i * (1-x)^(N-i)
#         A_upper[:, i] = comb(N, i) * (x_norm_upper)**i * (1 - x_norm_upper)**(N-i)
#         A_lower[:, i] = comb(N, i) * (x_norm_lower)**i * (1 - x_norm_lower)**(N-i)

#     # 使用最小二乘法拟合CST系数
#     CST_coefficients_upper = np.linalg.lstsq(A_upper * C_upper[:, np.newaxis], 
#                                             z_modified_upper, rcond=None)[0]
#     CST_coefficients_lower = np.linalg.lstsq(A_lower * C_lower[:, np.newaxis], 
#                                             z_modified_lower, rcond=None)[0]

#     # 计算拟合值时重新加上尾缘厚度项
#     z_fit_upper = np.dot(A_upper * C_upper[:, np.newaxis], CST_coefficients_upper) + x_norm_upper * z_te_upper
#     z_fit_lower = np.dot(A_lower * C_lower[:, np.newaxis], CST_coefficients_lower) + x_norm_lower * z_te_lower

#     # 计算拟合误差
#     def calculate_fitting_error(x_orig, z_orig, z_fit):
#         """
#         计算拟合误差
#         公式7: fitting_error = sum(w * |z_i - z_fit(x_i)|)
#         公式8: w = 2 if x_i < 0.2 else 1
#         """
#         # 计算权重（根据x坐标）
#         weights = np.where(x_orig < 0.2 * max_x, 2.0, 1.0)  # 注意这里使用原始x坐标判断
        
#         # 计算绝对误差并加权
#         absolute_errors = np.abs(z_orig - z_fit)
#         weighted_errors = weights * absolute_errors
        
#         # 总拟合误差
#         total_error = np.sum(weighted_errors)
        
#         return total_error, weighted_errors

#     # 计算上下表面的拟合误差
#     fitting_error_upper, weighted_errors_upper = calculate_fitting_error(x_upper, z_upper, z_fit_upper)
#     fitting_error_lower, weighted_errors_lower = calculate_fitting_error(x_lower, z_lower, z_fit_lower)
    
#     # 总拟合误差
#     total_fitting_error = fitting_error_upper + fitting_error_lower

#     # 输出误差信息
#     print(f"上表面拟合误差: {fitting_error_upper:.6f}")
#     print(f"下表面拟合误差: {fitting_error_lower:.6f}")
#     print(f"总拟合误差: {total_fitting_error:.6f}")
#     print(f"CST阶数 N = {N}")

#     return CST_coefficients_upper, CST_coefficients_lower, fitting_error_upper, fitting_error_lower, total_fitting_error

# # 调用函数（测试不同阶数）
# orders = [6, 8, 10]

# for order in orders:
#     print(f"\n{'='*50}")
#     print(f"测试CST阶数: {order}")
#     print(f"{'='*50}")
    
#     try:
#         # 请替换为您的实际数据文件路径
#         CST_coefficients_upper, CST_coefficients_lower, mse_upper, mse_lower, total_error = \
#             CST_parameterization_with_TE_thickness('D:\\airfoil_data\\processed_airfoils\\train\\ag04.csv', order)  # 假设数据文件为CSV格式
        
#         print(f"阶数 {order} 的拟合完成")
#         print(f"上表面CST系数数量: {len(CST_coefficients_upper)}")
#         print(f"下表面CST系数数量: {len(CST_coefficients_lower)}")
        
#     except Exception as e:
#         print(f"阶数 {order} 拟合失败: {e}")