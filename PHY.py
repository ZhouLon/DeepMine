import pandas as pd
import os
import glob
"""把所有的PHY的csv融合到一个文件夹中"""
# 设置工作目录到包含CSV文件的文件夹
os.chdir('./data/origin')  # 请替换为你的CSV文件所在的路径

# 获取所有CSV文件
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# 合并所有文件
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], ignore_index=True)

# 保存合并后的CSV文件
combined_csv.to_csv("fallPHY.csv", index=False, encoding='utf-8-sig')

