
import pandas as pd
import numpy as np

import csv
#首先运行tape-embed unirep ./data/origin/fPETase.fasta ./data/output/fPETase.npz babbler-1900 --tokenizer unirep --batch_size 512获得npz文件
#然后运行R得outputPHY
filename='fx.x.x.x'
arrays = np.load(f'./data/output/{filename}.npz', allow_pickle=True)
new_list={}

#下面需要归一化
for name,i in arrays.items():
    name_index=name.split('|')
    label = name_index[3]
    name=name_index[1]
    i=i.tolist()['avg']#转化为字典形式后，提取其中的avg键
    i = np.reshape(i, (1, -1))
    i-=np.min(i)
    i/=(np.max(i)-np.min(i))
    i = np.around(i, 4)
    new_list[name]=[label,name]+i[0].tolist()





PHY=pd.read_csv(f'./data/origin/{filename}PHY.csv')
PHY_arr=np.array(PHY)
for x in PHY_arr:
    name_index=x[0].split('|')
    name=name_index[1]
    label = name_index[3]
    PHY_4=x[1:].tolist()
    try:
        new_list[name]=new_list[name]+PHY_4
    except Exception as e:
        print(f"ERROR:{e}")
        print(name, label, PHY_4)

with open(f'./data/output/{filename}TAPE.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_list.values())


