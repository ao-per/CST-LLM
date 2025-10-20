import re
import os

def auto_parse_airfoil(file_path):
    """
    优化版：自动识别翼型文件格式并分离上下表面
    支持格式：
    1. 含点数标注（如"18. 18."、"40.0 41.0"，上下表面分行记录）
    2. 单段连续记录（无点数标注，坐标按"后缘→前缘→后缘"排列）
    :param file_path: 翼型.dat文件路径
    :return: 字典 {"UpX": [], "UpY": [], "LowX": [], "LowY": []}
    """
    # 读取文件内容并预处理
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        raise ValueError(f"文件{file_path}为空或无法读取")
    
    # -------------------------- 阶段1：优化点数行识别 --------------------------
    count_line_idx = -1
    n_up = n_low = 0
    # 匹配更广泛的点数格式（支持"18. 18."、"40.0 41.0"、" 20  21 "等）
    count_pattern = re.compile(r'\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s*')  # 支持整数和小数
    
    for i, line in enumerate(lines):
        match = count_pattern.fullmatch(line)
        if match:
            try:
                # 提取数字并转为整数（支持40.0→40）
                num1 = float(match.group(1))
                num2 = float(match.group(2))
                if num1.is_integer() and num2.is_integer():  # 确保是整数点数
                    n_up, n_low = int(num1), int(num2)
                    count_line_idx = i
                    break
            except:
                continue  # 跳过无法转换的行
    
    if count_line_idx != -1:
        # -------------------------- 处理格式1：含点数标注 --------------------------
        coord_lines = lines[count_line_idx + 1:]  # 从点数行后开始提取坐标
        coords = []
        for line in coord_lines:
            # 提取x和y（兼容正负值、科学计数法）
            nums = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*|[-+]?\d+', line)
            if len(nums) >= 2:  # 确保至少有x和y
                try:
                    x = float(nums[0])
                    y = float(nums[1])
                    coords.append((x, y))
                except:
                    continue
        
        # 点数验证（允许±1误差，兼容空行干扰）
        total_expected = n_up + n_low
        if not (total_expected - 1 <= len(coords) <= total_expected + 1):
            raise ValueError(
                f"格式1点数不匹配：预期{total_expected}个，实际{len(coords)}个"
            )
        
        # 严格按标注点数分割
        up_surface = coords[:n_up]
        low_surface = coords[n_up:n_up + n_low]  # 避免超出下表面点数
    
    else:
        # -------------------------- 处理格式2：单段连续记录 --------------------------
        coords = []
        for line in lines[1:]:  # 跳过首行名称
            nums = re.findall(r'[-+]?\d*\.\d+[eE]?[-+]?\d*|[-+]?\d+', line)
            if len(nums) >= 2:
                try:
                    x = float(nums[0])
                    y = float(nums[1])
                    coords.append((x, y))
                except:
                    continue
        
        if not coords:
            raise ValueError("未提取到有效坐标")
        
        # 寻找前缘点（x≈0.0）
        leading_edge_idx = None
        min_x = min(p[0] for p in coords)
        # 优先找x=0，找不到则用x最小的点
        for i, (x, y) in enumerate(coords):
            if abs(x) < 1e-6 or abs(x - min_x) < 1e-6:
                leading_edge_idx = i
                break
        
        # 分割上下表面
        up_surface = coords[:leading_edge_idx + 1]
        low_surface = coords[leading_edge_idx:]
    
    # -------------------------- 统一排序与去重 --------------------------
    # 按x升序排序（前缘→后缘）
    up_surface.sort(key=lambda p: p[0])
    low_surface.sort(key=lambda p: p[0])
    
    # 去重（移除x相同的重复点）
    def deduplicate(points):
        unique = []
        seen_x = set()
        for x, y in points:
            x_rounded = round(x, 6)  # 保留6位小数去重
            if x_rounded not in seen_x:
                seen_x.add(x_rounded)
                unique.append((x, y))
        return unique
    
    up_surface = deduplicate(up_surface)
    low_surface = deduplicate(low_surface)
    
    return {
        "UpX": [p[0] for p in up_surface],
        "UpY": [p[1] for p in up_surface],
        "LowX": [p[0] for p in low_surface],
        "LowY": [p[1] for p in low_surface]
    }

# -------------------------- 批量处理文件夹示例 --------------------------
def batch_process_airfoils(input_dir, output_dir=None):
    """
    批量处理文件夹中所有翼型文件，优化错误提示和结果保存
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 支持已存在文件夹
    
    success_count = 0
    fail_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.dat'):
            file_path = os.path.join(input_dir, filename)
            try:
                surfaces = auto_parse_airfoil(file_path)
                success_count += 1
                print(
                    f"✅ 处理成功：{filename} "
                    f"| 上表面：{len(surfaces['UpX'])}点 "
                    f"| 下表面：{len(surfaces['LowX'])}点"
                )
                
                if output_dir:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("UpX,UpY\n")
                        for x, y in zip(surfaces['UpX'], surfaces['UpY']):
                            f.write(f"{x:.6f},{y:.6f}\n")
                        f.write("\nLowX,LowY\n")
                        for x, y in zip(surfaces['LowX'], surfaces['LowY']):
                            f.write(f"{x:.6f},{y:.6f}\n")
            except Exception as e:
                fail_count += 1
                print(f"❌ 处理失败：{filename} | 错误：{str(e)}")
    
    print(f"\n处理完成：成功{success_count}个，失败{fail_count}个")

# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # 替换为实际文件夹路径
    input_folder = "D:\\airfoil_data\\test"
    output_folder = "D:\\airfoil_data\\processed_airfoils"
    
    batch_process_airfoils(input_folder, output_folder)


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


    