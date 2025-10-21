


# 实现1500个数据文件划分
import os
import shutil
import random
from glob import glob

def split_files_to_train_test(source_dir, train_dir, test_dir, total_files=1500, train_count=1000, test_count=500):
    """
    从源文件夹中随机选择指定数量的文件，按比例分配到train和test文件夹
    :param source_dir: 源文件所在文件夹
    :param train_dir: 训练集保存文件夹
    :param test_dir: 测试集保存文件夹
    :param total_files: 总共选择的文件数量（默认1500）
    :param train_count: 训练集文件数量（默认1000）
    :param test_count: 测试集文件数量（默认500）
    """
    # 校验数量合理性
    if train_count + test_count != total_files:
        raise ValueError(f"训练集数量({train_count}) + 测试集数量({test_count}) 不等于总数量({total_files})")
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取源文件夹中所有文件（仅保留文件，排除文件夹）
    all_files = glob(os.path.join(source_dir, '*'))
    all_files = [f for f in all_files if os.path.isfile(f)]
    
    # 检查源文件数量是否充足
    if len(all_files) < total_files:
        raise ValueError(f"源文件夹文件数量不足，需要{total_files}个，实际只有{len(all_files)}个")
    
    # 随机选择1500个文件
    selected_files = random.sample(all_files, total_files)
    
    # 打乱选中的文件顺序（增加随机性）
    random.shuffle(selected_files)
    
    # 分割为训练集（1000个）和测试集（500个）
    train_files = selected_files[:train_count]
    test_files = selected_files[train_count:]
    
    # 复制到训练集
    for file in train_files:
        filename = os.path.basename(file)
        dest_path = os.path.join(train_dir, filename)
        shutil.copy2(file, dest_path)  # 保留文件元数据
    
    # 复制到测试集
    for file in test_files:
        filename = os.path.basename(file)
        dest_path = os.path.join(test_dir, filename)
        shutil.copy2(file, dest_path)
    
    print(f"分割完成：共选择{total_files}个文件")
    print(f"训练集：{len(train_files)}个文件（保存至{train_dir}）")
    print(f"测试集：{len(test_files)}个文件（保存至{test_dir}）")

# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":  
    # 配置路径（替换为你的实际路径）
    source_folder = "D:\\airfoil_data\\processed_airfoils"  # 源文件所在文件夹
    train_folder = "D:\\airfoil_data\\processed_airfoils\\train"  # 训练集路径
    test_folder = "D:\\airfoil_data\\processed_airfoils\\test"    # 测试集路径
    
    # 按1000:500分割（总共1500个文件）
    split_files_to_train_test(
        source_dir=source_folder,
        train_dir=train_folder,
        test_dir=test_folder,
        total_files=1500,
        train_count=1000,
        test_count=500
    )













# 实现两种dat文件的处理得到upx，upy，lowx，lowy等数据
# import re
# import os


# def auto_parse_airfoil(file_path):
#     """
#     优化版：自动识别翼型文件格式并分离上下表面
#     支持格式：
#     1. 含点数标注（如"18. 18."、"40.0 41.0"，上下表面分行记录）
#     2. 单段连续记录（无点数标注，坐标按"后缘→前缘→后缘"排列）
#     :param file_path: 翼型.dat文件路径
#     :return: 字典 {"UpX": [], "UpY": [], "LowX": [], "LowY": []}
#     """
#     # 读取文件内容并预处理：保留原始空格结构，仅过滤空行
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]
    
#     if not lines:
#         raise ValueError(f"文件{file_path}为空或无法读取")
    
#     # -------------------------- 阶段1：优化点数行识别 --------------------------
#     count_line_idx = -1
#     n_up = n_low = 0
#     # 匹配更广泛的点数格式（支持整数、小数，且点数行无其他字符）
#     count_pattern = re.compile(r'^\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s*$')
    
#     for i, line in enumerate(lines):
#         match = count_pattern.fullmatch(line)
#         if match:
#             try:
#                 num1 = float(match.group(1))
#                 num2 = float(match.group(2))
#                 # 点数需为正整数（排除0或负数）
#                 if num1.is_integer() and num2.is_integer() and num1 > 0 and num2 > 0:
#                     n_up, n_low = int(num1), int(num2)
#                     count_line_idx = i
#                     break
#             except:
#                 continue  # 跳过无法转换的非点数行
    
#     if count_line_idx != -1:
#         # -------------------------- 处理格式1：含点数标注 --------------------------
#         coord_lines = lines[count_line_idx + 1:]
#         coords = []
#         for line in coord_lines:
#             # 提取x/y：兼容正负、科学计数法，且仅取前两个有效数值（排除多余列干扰）
#             nums = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*|[-+]?\d+', line)
#             if len(nums) >= 2:
#                 try:
#                     coords.append((float(nums[0]), float(nums[1])))
#                 except:
#                     continue
        
#         total_expected = n_up + n_low
#         if not (total_expected - 2 <= len(coords) <= total_expected + 2):
#             raise ValueError(
#                 f"格式1点数不匹配：预期{total_expected}个，实际{len(coords)}个"
#             )
        
#         up_surface = coords[:n_up]
#         low_surface = coords[n_up:n_up + n_low]
    
#     else:
#         # -------------------------- 处理格式2：单段连续记录（重点修复） --------------------------
#         # 步骤1：提取坐标（首行是翼型名称/信息，从第2行开始；兼容首行即坐标的极端情况）
#         coords = []
#         # 先过滤非坐标行（排除首行可能的名称、注释，仅保留含2个数值的行）
#         coord_candidate_lines = []
#         for line in lines:
#             nums = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*|[-+]?\d+', line)
#             if len(nums) >= 2:
#                 coord_candidate_lines.append(line)
        
#         # 提取坐标
#         for line in coord_candidate_lines:
#             nums = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*|[-+]?\d+', line)
#             try:
#                 coords.append((float(nums[0]), float(nums[1])))
#             except:
#                 continue
        
#         if not coords:
#             raise ValueError("未提取到有效坐标数据")
        
#         # 步骤2：精准定位前缘点（核心修复：x最小且y接近0的点，解决E184前缘点非x=0问题）
#         # 先找x最小的点集合（允许±1e-6误差）
#         min_x = min(p[0] for p in coords)
#         leading_candidates = [
#             (i, x, y) for i, (x, y) in enumerate(coords) 
#             if abs(x - min_x) < 1e-6
#         ]
        
#         if not leading_candidates:
#             raise ValueError("无法找到x最小的前缘候选点")
        
#         # 从候选点中选y最接近0的点（翼型前缘y通常接近0，排除异常点）
#         leading_candidates.sort(key=lambda item: abs(item[2]))  # 按y绝对值排序
#         leading_edge_idx = leading_candidates[0][0]  # 取y最接近0的点的索引
        
#         # 步骤3：分割初始上下表面（按"后缘→前缘→后缘"的顺序分割）
#         up_surface = coords[:leading_edge_idx + 1]  # 从起始（后缘）到前缘
#         low_surface = coords[leading_edge_idx:]     # 从前缘到结束（后缘）
        
#         # 步骤4：上下表面正确性校验与修正（解决E184上表面初始y负的问题）
#         # 规则1：上表面应包含"后缘（x≈1）→前缘"的完整段，需包含x≈1的点
#         has_up_trailing = any(abs(p[0] - 1.0) < 1e-3 for p in up_surface)
#         has_low_trailing = any(abs(p[0] - 1.0) < 1e-3 for p in low_surface)
        
#         # 若上表面没有后缘点（x≈1），说明分割反了，交换上下表面
#         if not has_up_trailing and has_low_trailing:
#             up_surface, low_surface = low_surface, up_surface
        
#         # 规则2：通过"最大y值"判断（上表面最大y值应显著大于下表面）
#         max_up_y = max(p[1] for p in up_surface) if up_surface else -float('inf')
#         max_low_y = max(p[1] for p in low_surface) if low_surface else -float('inf')
        
#         # 若上表面最大y小于下表面，交换（确保上表面是"凸"的一面）
#         if max_up_y < max_low_y - 1e-6:  # 加1e-6避免浮点误差
#             up_surface, low_surface = low_surface, up_surface
    
#     # -------------------------- 统一后处理：排序、去重、过滤 --------------------------
#     # 按x升序排序（确保从前缘→后缘的顺序，便于后续使用）
#     up_surface.sort(key=lambda p: p[0])
#     low_surface.sort(key=lambda p: p[0])
    
#     # 去重：移除x相同的重复点（保留第一个，避免后续计算异常）
#     def deduplicate(points):
#         unique = []
#         seen_x = set()
#         for x, y in points:
#             x_rounded = round(x, 6)  # 保留6位小数，平衡精度与去重效果
#             if x_rounded not in seen_x:
#                 seen_x.add(x_rounded)
#                 unique.append((x, y))
#         return unique
    
#     up_surface = deduplicate(up_surface)
#     low_surface = deduplicate(low_surface)
    
#     # 过滤异常点：移除y值超出合理范围的点（翼型y通常在-0.2~0.2之间，可根据需求调整）
#     def filter_abnormal(points):
#         return [
#             (x, y) for x, y in points 
#             if -0.2 <= y <= 0.2
#         ]
    
#     up_surface = filter_abnormal(up_surface)
#     low_surface = filter_abnormal(low_surface)
    
#     # 最终校验：确保上下表面非空
#     if not up_surface:
#         raise ValueError("处理后上表面无有效坐标")
#     if not low_surface:
#         raise ValueError("处理后下表面无有效坐标")
    
#     return {
#         "UpX": [p[0] for p in up_surface],
#         "UpY": [p[1] for p in up_surface],
#         "LowX": [p[0] for p in low_surface],
#         "LowY": [p[1] for p in low_surface]
#     }

# # -------------------------- 批量处理文件夹 --------------------------
# def batch_process_airfoils(input_dir, output_dir=None):
#     """
#     批量处理文件夹中所有.dat翼型文件，输出CSV格式的上下表面坐标
#     :param input_dir: 输入文件夹路径（存放.dat文件）
#     :param output_dir: 输出文件夹路径（存放.csv文件，None则不保存）
#     """
#     # 验证输入文件夹存在
#     if not os.path.exists(input_dir):
#         raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")
    
#     # 创建输出文件夹（若指定）
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
    
#     success_count = 0
#     fail_count = 0
#     fail_files = []  # 记录失败文件，便于后续排查
    
#     # 遍历所有.dat文件
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith('.dat'):
#             file_path = os.path.join(input_dir, filename)
#             try:
#                 # 解析翼型
#                 surfaces = auto_parse_airfoil(file_path)
#                 success_count += 1
#                 # 打印成功信息（包含关键参数，便于快速确认）
#                 print(
#                     f"✅ 成功：{filename:20} "
#                     f"上表面{len(surfaces['UpX']):3}点 "
#                     f"下表面{len(surfaces['LowX']):3}点 "
#                     f"上表面最大y：{max(surfaces['UpY']):.4f}"
#                 )
                
#                 # 保存CSV（若指定输出目录）
#                 if output_dir:
#                     output_filename = f"{os.path.splitext(filename)[0]}.csv"
#                     output_path = os.path.join(output_dir, output_filename)
#                     with open(output_path, 'w', encoding='utf-8') as f:
#                         # 写入上表面
#                         f.write("UpX,UpY\n")
#                         for x, y in zip(surfaces['UpX'], surfaces['UpY']):
#                             f.write(f"{x:.6f},{y:.6f}\n")
#                         # 空行分隔，便于阅读
#                         f.write("\n")
#                         # 写入下表面
#                         f.write("LowX,LowY\n")
#                         for x, y in zip(surfaces['LowX'], surfaces['LowY']):
#                             f.write(f"{x:.6f},{y:.6f}\n")
            
#             except Exception as e:
#                 fail_count += 1
#                 fail_files.append((filename, str(e)))
#                 print(f"❌ 失败：{filename:20} 错误：{str(e)[:50]}...")  # 截断长错误信息
    
#     # 打印处理总结
#     print("\n" + "="*60)
#     print(f"处理总结：共{success_count + fail_count}个文件")
#     print(f"✅ 成功：{success_count}个")
#     print(f"❌ 失败：{fail_count}个")
#     if fail_files:
#         print("\n失败文件详情：")
#         for fn, err in fail_files:
#             print(f"  {fn}: {err}")

# # -------------------------- 使用示例 --------------------------
# if __name__ == "__main__":
#     # 请替换为你的实际路径（Windows用双反斜杠，Linux/macOS用单斜杠）
#     INPUT_FOLDER = "D:\\airfoil_data\\test"    # 存放E184.dat等文件的文件夹
#     OUTPUT_FOLDER = "D:\\airfoil_data\\processed_airfoils"  # 输出CSV的文件夹
    
#     # 执行批量处理
#     batch_process_airfoils(INPUT_FOLDER, OUTPUT_FOLDER)


#D:\\airfoil_data\\train\\a18.dat
#D:\\airfoil_data\\train








#第一种数据：a18.dat
# import re

# def split_airfoil_single_segment(file_content):
#     """
#     处理单段连续记录的翼型数据（x从1→0→1，自动分离上下表面）
#     :param file_content: 翼型文件内容（字符串）
#     :return: 上下表面坐标字典
#     """
#     # 1. 提取坐标数据（跳过首行名称）
#     lines = [line.strip() for line in file_content.splitlines() if line.strip()]
#     coords = []
#     for line in lines[1:]:  # 跳过首行（如"A18 (original)"）
#         nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
#         if len(nums) == 2:
#             x, y = float(nums[0]), float(nums[1])
#             coords.append((x, y))
    
#     # 2. 找到x=0的前缘点（分割点）
#     # 坐标顺序：后缘(1.0)→前缘(0.0)→后缘(1.0)，以x=0为界
#     leading_edge_idx = None
#     for i, (x, y) in enumerate(coords):
#         if abs(x - 0.0) < 1e-6:  # 找到x≈0的点
#             leading_edge_idx = i
#             break
#     if leading_edge_idx is None:
#         raise ValueError("未找到前缘点（x=0.0）")
    
#     # 3. 分割上下表面：
#     # - 前缘点之前（x从1→0）：上表面（Y值较大）
#     # - 前缘点之后（x从0→1）：下表面（Y值较小）
#     up_surface = coords[:leading_edge_idx + 1]  # 包含前缘点
#     low_surface = coords[leading_edge_idx:]     # 包含前缘点（与上表面共享）
    
#     # 4. 按x升序排序（统一为前缘→后缘）
#     up_surface.sort(key=lambda p: p[0])
#     low_surface.sort(key=lambda p: p[0])
    
#     # 提取x和y列表
#     return {
#         "UpX": [p[0] for p in up_surface],
#         "UpY": [p[1] for p in up_surface],
#         "LowX": [p[0] for p in low_surface],
#         "LowY": [p[1] for p in low_surface]
#     }

# # -------------------------- 测试示例 --------------------------
# if __name__ == "__main__":
#     # 输入A18翼型数据
#     with open("D:\\airfoil_data\\train\\a18.dat", "r") as f: airfoil_data = f.read()
#     # 分离上下表面
#     surfaces = split_airfoil_single_segment(airfoil_data)
    
#     # 输出结果
#     print("上表面（UpX, UpY）：")
#     for x, y in zip(surfaces["UpX"], surfaces["UpY"]):
#         print(f"{x:.5f}  {y:.5f}")
    
#     print("\n下表面（LowX, LowY）：")
#     for x, y in zip(surfaces["LowX"], surfaces["LowY"]):
#         print(f"{x:.5f}  {y:.5f}")



# Args:
#         foilpath (str): 翼型文件(.dat)绝对路径

#     Returns:
#         FoilData (Dict):
#             "Name" (str): 翼型名称
#             "Chord" (float): 翼型弦长
#             "Num" (int): 翼型坐标点个数
#             "UpX" (list): 翼型上表面x坐标
#             "UpY" (list): 翼型上表面Y坐标
#             "LowX" (list): 翼型下表面x坐标
#             "LowY" (list): 翼型下表面y坐标














#第二种数据：2023c.dat
# import re

# def split_airfoil_by_count(file_content):
#     """
#     根据数据中指定的点数（18上表面+18下表面）分离坐标
#     :param file_content: 翼型文件内容（字符串）
#     :return: 上下表面坐标字典
#     """
#     # 按行处理，过滤空行
#     lines = [line.strip() for line in file_content.splitlines() if line.strip()]
    
#     # 1. 提取点数信息（寻找"18. 18."所在行）
#     n_up = 0
#     n_low = 0
#     count_line_idx = -1
#     for i, line in enumerate(lines):
#         # 匹配类似"18. 18."的行（允许空格和小数点差异）
#         if re.match(r'\s*\d+\.\s+\d+\.\s*', line):
#             nums = re.findall(r'\d+', line)
#             if len(nums) == 2:
#                 n_up, n_low = int(nums[0]), int(nums[1])  # 18, 18
#                 count_line_idx = i
#                 break
#     if count_line_idx == -1:
#         raise ValueError("未找到点数信息行（如'18. 18.'）")
    
#     # 2. 提取坐标数据（从点数行之后开始）
#     coord_lines = lines[count_line_idx+1:]  # 点数行之后的所有行
#     coords = []
#     for line in coord_lines:
#         nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
#         if len(nums) == 2:
#             x, y = float(nums[0]), float(nums[1])
#             coords.append((x, y))
    
#     # 3. 按点数分割：前18个为上表面，后18个为下表面
#     if len(coords) != n_up + n_low:
#         raise ValueError(f"实际点数与指定不符（需{ n_up + n_low }个，实际{ len(coords) }个）")
    
#     up_surface = coords[:n_up]  # 上表面（前18个点）
#     low_surface = coords[n_up:]  # 下表面（后18个点）
    
#     # 4. 按x升序排序（确保从前缘到后缘）
#     up_surface.sort(key=lambda p: p[0])
#     low_surface.sort(key=lambda p: p[0])
    
#     # 提取x和y列表
#     return {
#         "UpX": [p[0] for p in up_surface],
#         "UpY": [p[1] for p in up_surface],
#         "LowX": [p[0] for p in low_surface],
#         "LowY": [p[1] for p in low_surface]
#     }

# # -------------------------- 测试示例 --------------------------
# if __name__ == "__main__":
#     # 输入你的翼型数据
#     with open("D:\\airfoil_data\\train\\2032c.dat", "r") as f: airfoil_data = f.read()

    
#     # 分离上下表面
#     surfaces = split_airfoil_by_count(airfoil_data)
    
#     # 输出结果（验证点数是否各为18）
#     print(f"上表面点数：{len(surfaces['UpX'])}（应为18）")
#     print("UpX:", surfaces["UpX"])
#     print("UpY:", surfaces["UpY"])
    
#     print(f"\n下表面点数：{len(surfaces['LowX'])}（应为18）")
#     print("LowX:", surfaces["LowX"])
#     print("LowY:", surfaces["LowY"])


    