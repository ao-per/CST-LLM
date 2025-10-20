import os
import requests
from bs4 import BeautifulSoup
import time
import re

def create_directory():
    """创建存储目录"""
    directory = "D:/airfoil_data"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")
    return directory

def get_airfoil_links():
    """从网页内容中提取所有.dat文件的链接"""
    url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 查找所有.dat文件的链接
        dat_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.dat'):
                # 处理相对链接
                if href.startswith('http'):
                    full_url = href
                else:
                    full_url = f"https://m-selig.ae.illinois.edu/ads/{href}"
                dat_links.append(full_url)
        
        return list(set(dat_links))  # 去重
    
    except Exception as e:
        print(f"获取链接时出错: {e}")
        return []

def download_airfoil_file(url, save_dir):
    """下载单个.dat文件"""
    try:
        # 从URL中提取文件名
        filename = url.split('/')[-1]
        filepath = os.path.join(save_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(filepath):
            print(f"文件已存在，跳过: {filename}")
            return True
        
        # 下载文件
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存文件
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"成功下载: {filename}")
        return True
        
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def main():
    """主函数"""
    print("开始下载UIUC翼型数据库...")
    
    # 创建目录
    save_dir = create_directory()
    
    # 获取所有.dat文件链接
    print("正在获取文件列表...")
    dat_links = get_airfoil_links()
    
    if not dat_links:
        print("未找到任何.dat文件链接")
        return
    
    print(f"找到 {len(dat_links)} 个.dat文件")
    
    # 下载文件
    success_count = 0
    failed_count = 0
    
    for i, link in enumerate(dat_links, 1):
        print(f"正在下载第 {i}/{len(dat_links)} 个文件...")
        
        if download_airfoil_file(link, save_dir):
            success_count += 1
        else:
            failed_count += 1
        
        # 添加延迟避免请求过快
        time.sleep(0.5)
    
    # 统计结果
    print("\n" + "="*50)
    print("下载完成！")
    print(f"存储目录: {save_dir}")
    print(f"成功下载: {success_count} 个文件")
    print(f"下载失败: {failed_count} 个文件")
    print(f"总计找到: {len(dat_links)} 个文件")
    
    # 保存下载记录
    record_file = os.path.join(save_dir, "download_record.txt")
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write(f"下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"成功下载: {success_count} 个文件\n")
        f.write(f"下载失败: {failed_count} 个文件\n")
        f.write(f"总计找到: {len(dat_links)} 个文件\n")
        f.write(f"存储路径: {save_dir}\n")
    
    print(f"下载记录已保存至: {record_file}")

if __name__ == "__main__":
    main()