from Bio import SeqIO

# 打开FASTA文件
fasta_file = "data/filtered_3.1.1.x-4199.fasta"  #替换成要检查的文件路径，fasta总文件

# 使用SeqIO.parse读取文件中的每个序列
non_list=[]
num=0
for record in SeqIO.parse(fasta_file, "fasta"):
    num+=1
    # 打印序列的ID和序列长度
    print(f"Sequence ID: {record.id}")
    print(f"Sequence Length: {len(record.seq)}")
    print(f"Sequence: {record.seq}")
    for i in record.seq:
        if i not in 'ACDEFGHIKLMNPQRSTVWY':
            print('non')
            non_list.append(record.id)
            break
    else:
        print('yes')
print(non_list,num,len(non_list))


