import numpy as np
from scipy.special import comb
import os

def calculate_fitting_error(x_upper, z_upper, x_lower, z_lower, 
                           z_fit_upper, z_fit_lower, max_x):
    """计算翼型拟合误差"""
    weights_upper = np.where(x_upper < 0.2 * max_x, 2.0, 1.0)
    error_upper = np.sum(weights_upper * np.abs(z_upper - z_fit_upper))
    
    weights_lower = np.where(x_lower < 0.2 * max_x, 2.0, 1.0)
    error_lower = np.sum(weights_lower * np.abs(z_lower - z_fit_lower))
    
    return error_upper, error_lower, error_upper + error_lower

def CST_fitting_error(file_path, N):
    """仅计算并打印拟合误差"""
    # 加载数据
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 解析数据
    upper, lower = [], []
    section = None
    for line in lines:
        if line.startswith('UpX,UpY'):
            section = 'upper'
            continue
        elif line.startswith('LowX,LowY'):
            section = 'lower'
            continue
        
        if section == 'upper':
            x, y = map(float, line.split(','))
            upper.append([x, y])
        elif section == 'lower':
            x, y = map(float, line.split(','))
            lower.append([x, y])
    
    upper = np.array(upper)
    lower = np.array(lower)
    
    # 准备数据（上表面降序，下表面升序）
    x_upper = upper[:, 0][::-1]
    z_upper = upper[:, 1][::-1]
    x_lower = lower[:, 0]
    z_lower = lower[:, 1]
    
    # 归一化
    max_x = np.max(x_upper)
    x_norm_upper = x_upper / max_x
    x_norm_lower = x_lower / max_x
    
    # 尾缘厚度
    z_te_upper = z_upper[0]
    z_te_lower = z_lower[-1]
    
    # 修正z坐标
    z_modified_upper = z_upper - x_norm_upper * z_te_upper
    z_modified_lower = z_lower - x_norm_lower * z_te_lower
    
    # CST参数化
    N1, N2 = 0.5, 1.0
    C_upper = (x_norm_upper)**N1 * (1 - x_norm_upper)**N2
    C_lower = (x_norm_lower)**N1 * (1 - x_norm_lower)**N2
    
    # 伯恩斯坦多项式
    A_upper = np.array([[comb(N, i) * (x**i) * ((1-x)**(N-i)) 
                       for i in range(N+1)] for x in x_norm_upper])
    A_lower = np.array([[comb(N, i) * (x**i) * ((1-x)**(N-i)) 
                       for i in range(N+1)] for x in x_norm_lower])
    
    # 最小二乘拟合
    coeff_upper = np.linalg.lstsq(A_upper * C_upper[:, None], z_modified_upper, rcond=None)[0]
    coeff_lower = np.linalg.lstsq(A_lower * C_lower[:, None], z_modified_lower, rcond=None)[0]
    
    # 计算拟合值
    z_fit_upper = (A_upper * C_upper[:, None]) @ coeff_upper + x_norm_upper * z_te_upper
    z_fit_lower = (A_lower * C_lower[:, None]) @ coeff_lower + x_norm_lower * z_te_lower
    
    return calculate_fitting_error(x_upper, z_upper, x_lower, z_lower,
                                z_fit_upper, z_fit_lower, max_x)

def process_folder(folder_path, output_file):
    """批量处理文件夹内的所有CSV文件"""
    orders = [6, 8, 10]
    total_results = []  # 存储所有文件的结果
    
    # 获取文件夹内所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
with open(output_file, 'w', encoding='utf-8') as f_out:  # 关键修改
    # 写入表头
    f_out.write("文件名\t阶数6上表面\t阶数6下表面\t阶数6总误差\t"
                "阶数8上表面\t阶数8下表面\t阶数8总误差\t"
                "阶数10上表面\t阶数10下表面\t阶数10总误差\t"
                "三阶总和\n")
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        file_results = [csv_file]  # 存储当前文件的结果
        
        try:
            sum_errors = 0  # 用于累加6,8,10阶的总误差
            order_errors = []  # 存储各阶误差
            
            for order in orders:
                err_up, err_low, total_err = CST_fitting_error(file_path, order)
                order_errors.extend([err_up, err_low, total_err])
                sum_errors += total_err
            
            # 将结果写入列表
            file_results.extend([f"{x:.6f}" for x in order_errors])
            file_results.append(f"{sum_errors:.6f}")
            
            # 写入文件（制表符分隔）
            f_out.write("\t".join(file_results) + "\n")
            
            print(f"处理完成: {csv_file}")
            
        except Exception as e:
            print(f"处理失败 {csv_file}: {str(e)}")
            # 写入错误标记
            f_out.write(f"{csv_file}\tERROR\t{str(e)[:50]}\n")

if __name__ == "__main__":
    folder_path = r"D:\CST-LLM\processsed_airfoil_data\test"
    output_file="D:\\CST-LLM\\results.csv"
    process_folder(folder_path, output_file)