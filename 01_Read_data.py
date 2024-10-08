import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# File_dir=sys.argv[1]
# Read train file
global Train_dir,Valid_dir,Test_dir
## Give the number of random state
SEED=0
## Read file , X = path of dir , seed = the number of random_state 
def read_file(X,seed):
    File = pd.read_csv(str(X), header=None)
    File = np.array(File)
    File_X=File[:,2:]
    File_Y=File[:,0:2]
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(File_X, File_Y, test_size = 0.1, random_state = seed)
    Train_X, Valid_X, Train_Y, Valid_Y = train_test_split(Train_X, Train_Y, test_size = 0.1, random_state = seed)
    Train_Data=np.c_[Train_Y,Train_X]
    Valid_Data=np.c_[Valid_Y,Valid_X]
    Test_Data=np.c_[Test_Y,Test_X]
    return Train_Data , Valid_Data , Test_Data
## Give the dir path
# def make_file(path):
#     global Train_dir,Valid_dir,Test_dir
#     File_array=[]
#     for file_name in os.listdir(str(path)):
#         File_array.append(file_name)
#     Train_dir=str(path)+"/Train"
#     Valid_dir=str(path)+"/Valid"
#     Test_dir=str(path)+"/Test"
#     os.mkdir(Train_dir)
#     os.mkdir(Valid_dir)
#     os.mkdir(Test_dir)
#     for file_name in File_array:
#         file=str(path)+"/"+str(file_name)
#         Train_Data , Valid_Data , Test_Data=read_file(file,SEED)
#         with open(Train_dir+"/Train_"+str(file_name),"w") as Train_file:
#             np.savetxt(Train_file,Train_Data, fmt='%s',delimiter=',')
#         with open(Valid_dir+"/Valid_"+str(file_name),"w") as Valid_file:
#             np.savetxt(Valid_file,Valid_Data, fmt='%s',delimiter=',')
#         with open(Test_dir+"/Test_"+str(file_name),"w") as Test_file:
#             np.savetxt(Test_file,Test_Data, fmt='%s',delimiter=',')
# ## Put all train/valid/test data in one file , thr order should be Train Valid Test
# def sort_file():
#     num=0
#     Train_X=np.array([])
#     Valid_X=np.array([])
#     Test_X=np.array([])
#     Train_Y=np.array([])
#     Valid_Y=np.array([])
#     Test_Y=np.array([])
#     global Train_dir,Valid_dir,Test_dir
#     Total=[]
#     Dir=[Train_dir,Valid_dir,Test_dir]
#     X=[Train_X,Valid_X,Test_X]
#     Y=[Train_Y,Valid_Y,Test_Y]
#     for i in range(len(Dir)):
#         for file_name in os.listdir(str(Dir[i])):
#             File = pd.read_csv((str(Dir[i])+"/"+str(file_name)), header=None)
#             File = np.array(File)
#             File_X=File[:,2:]
#             File_Y=File[:,0:2]
#             File=np.c_[File_Y,File_X]
#             if num==0:
#                 with open(Train_dir+"/Total_Train.csv","a+") as Total_Train_file:
#                     np.savetxt(Total_Train_file,File, fmt='%s',delimiter=',')
#             if num==1:
#                 with open(Valid_dir+"/Total_Valid.csv","a+") as Total_Valid_file:
#                     np.savetxt(Total_Valid_file,File, fmt='%s',delimiter=',')
#             if num==2:
#                 with open(Test_dir+"/Total_Test.csv","a+") as Total_Test_file:
#                     np.savetxt(Total_Test_file,File, fmt='%s',delimiter=',')
#         num+=1

# make_file(File_dir)
# sort_file()
File="/share/lijianfeng/final/for_fu/tape_phy/70/VAL_70_tape_phy.csv"

Train_Data,Valid_Data,Test_Data=read_file(File,SEED)
with open("/share/lijianfeng/final/for_fu/tape_phy/70/train/VAL.csv","w") as Train_file:
    np.savetxt(Train_file,Train_Data, fmt='%s',delimiter=',')
with open("/share/lijianfeng/final/for_fu/tape_phy/70/valid/VAL.csv","w") as Valid_file:
    np.savetxt(Valid_file,Valid_Data, fmt='%s',delimiter=',')
with open("/share/lijianfeng/final/for_fu/tape_phy/70/test/VAL.csv","w") as Test_file:
    np.savetxt(Test_file,Test_Data, fmt='%s',delimiter=',')
print("Well Done")