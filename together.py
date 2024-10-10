import pandas as pd
import os

# 指定存放CSV文件的目录
directory = './data/output'

# 列出目录中所有的CSV文件
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# 初始化一个空的DataFrame列表
frames = []

# 遍历所有CSV文件
for csv_file in csv_files:
    # 读取CSV文件到DataFrame
    df = pd.read_csv(os.path.join(directory, csv_file))

    # 将DataFrame添加到列表中
    frames.append(df)

# 使用concat函数合并所有DataFrame
merged_df = pd.concat(frames, ignore_index=True)

# 保存合并后的DataFrame到新的CSV文件
merged_df.to_csv('ALL.csv', index=False)
