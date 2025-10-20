import os
import random
import shutil

def split_airfoil_data(source_dir="D:/airfoil_data", train_size=1000, test_size=500):
    """
    将翼型数据文件分割为训练集、测试集和其他文件
    
    参数:
        source_dir: 原始数据目录路径
        train_size: 训练集文件数量
        test_size: 测试集文件数量
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：目录 {source_dir} 不存在")
        return
    
    # 创建目标目录
    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")
    other_dir = os.path.join(source_dir, "other")
    
    for dir_path in [train_dir, test_dir, other_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有.dat文件
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.dat')]
    total_files = len(all_files)
    
    # 检查文件数量是否足够
    required_files = train_size + test_size
    if total_files < required_files:
        print(f"错误：目录中只有 {total_files} 个.dat文件，但需要至少 {required_files} 个")
        return
    
    # 随机打乱文件列表
    random.shuffle(all_files)
    
    # 分割文件
    train_files = all_files[:train_size]
    test_files = all_files[train_size:train_size + test_size]
    other_files = all_files[train_size + test_size:]
    
    # 移动文件到对应目录
    def move_files(files, dest_dir):
        for file in files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(dest_dir, file)
            shutil.move(src, dst)
    
    move_files(train_files, train_dir)
    move_files(test_files, test_dir)
    move_files(other_files, other_dir)
    
    # 打印统计信息
    print("文件分割完成:")
    print(f"- 训练集: {len(train_files)} 个文件 (存放在 {train_dir})")
    print(f"- 测试集: {len(test_files)} 个文件 (存放在 {test_dir})")
    print(f"- 其他文件: {len(other_files)} 个文件 (存放在 {other_dir})")
    
    # 保存分割记录
    record_file = os.path.join(source_dir, "split_record.txt")
    with open(record_file, 'w') as f:
        f.write("翼型数据文件分割记录\n")
        f.write(f"训练集文件数: {len(train_files)}\n")
        f.write(f"测试集文件数: {len(test_files)}\n")
        f.write(f"其他文件数: {len(other_files)}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write("\n训练集文件列表:\n")
        f.write("\n".join(train_files))
        f.write("\n\n测试集文件列表:\n")
        f.write("\n".join(test_files))
    
    print(f"\n分割记录已保存到: {record_file}")

if __name__ == "__main__":
    split_airfoil_data()