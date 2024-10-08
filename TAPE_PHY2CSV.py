
import pandas as pd
import numpy as np

import csv
#首先运行tape-embed unirep ./data/origin/fPETase.fasta ./data/output/many/fPETase.npz babbler-1900 --tokenizer unirep获得npz文件
#然后运行R得outputPHY
filename='test'
arrays = np.load(f'./data/output/many/{filename}.npz', allow_pickle=True)
new_list={}

#下面需要归一化
for name,i in arrays.items():
    label = name[-1]
    name=name[:name.index('-')]
    i=i.tolist()['avg']#转化为字典形式后，提取其中的avg键
    i = np.reshape(i, (1, -1))
    i-=np.min(i)
    i/=(np.max(i)-np.min(i))
    i = np.around(i, 4)
    new_list[name]=[label,name]+i[0].tolist()

# PHY=pd.read_csv(f'./data/many/{filename}PHY.csv')
# PHY_arr=np.array(PHY)
# for x in PHY_arr:
#     name=x[0]
#     label = name[-1]
#     name = name[:name.index('-')]
#
#     new_list[name]=new_list[name]+x[1:].tolist()



print(len(arrays.keys()))
with open(f'./data/output/final/{filename}TAPE.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_list.values())



