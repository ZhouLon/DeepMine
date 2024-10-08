import os
import timeit
import datetime
import math
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAvgPool3D, GlobalMaxPool3D
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.layers import concatenate, add, multiply, average, maximum
from tensorflow.keras.layers import BatchNormalization, Activation,concatenate
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from tensorflow.keras import optimizers,regularizers
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from pickle import load
import sys
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import os
from collections import Counter
from imblearn.over_sampling import SMOTE


os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Unirep First
# You should run this after running Read_data.py
File_dir="/share/lijianfeng/final/Data/Total_data/raw/"
def read_file(X):
    File = pd.read_csv(str(X), header=None)
    File = np.array(File)
    File_X=File[:,2:]
    File_Y=File[:,0]
    File_item_name=File[:,1]
    File_X=np.array(File_X)
    File_Y=np.array(File_Y)
    File_item_name=np.array(File_item_name)
    File_X=File_X.astype('float64')
    return File_X,File_Y,File_item_name
Train_X,Train_Y,Train_item_name=read_file(File_dir+"Train/Total_Train.csv")

Valid_X,Valid_Y,Valid_item_name=read_file(File_dir+"Valid/Total_Valid.csv")
Valid_X1,Valid_Y1,Valid_item_name1=read_file("/home/lijianfeng/Experment_article_99_nonredu_rename_TAPE_PHY.csv")
Test_X,Test_Y,Test_item_name=read_file(File_dir+"Test/Total_Test.csv")
min_phy=[0,0,0,0]
min_phy=np.array(min_phy)
max_phy=[0,0,0,0]
max_phy=np.array(max_phy)
##The data for each item should be [1,1900]
def sigmoid_function(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j]=(1/(1 + math.exp(-z[i][j])))
    return z
def Trans_data(Array):
    Unirep_array=[]
    PHY_array=[]
    Unirep_arr=Array[:,0:1900]
    Phy_arr=Array[:,1900:1904]
    PHY_arr=sigmoid_function(Phy_arr)
    print(Array.shape)
    print(Unirep_arr.shape)
    for i in range(len(Unirep_arr)):
        sort_array=np.reshape(Unirep_arr[i],(1,1900))
        Unirep_array.append(sort_array)
    Unirep_array=np.array(Unirep_array)
    Unirep_array=np.reshape(Unirep_array,(-1,19,25,4))
    Unirep_array=Unirep_array.astype('float64')  
    for i in range(len(PHY_arr)):
        sort_array=np.reshape(PHY_arr[i],(1,4))
        # C=np.zeros((1,4))
        # sort_array=np.c_[sort_array,C]
        # Total_array.append(sort_array)
        PHY_array.append(sort_array)
    PHY_array=np.array(PHY_array)
    PHY_array=np.reshape(PHY_array,(-1,1,2,2))
    PHY_array=PHY_array.astype('float64')
    return Unirep_array,PHY_array
Unirep_Train_data,PHY_Train_data=Trans_data(Train_X)
Unirep_Valid_data,PHY_Valid_data=Trans_data(Valid_X)
Unirep_Valid_data1,PHY_Valid_data1=Trans_data(Valid_X1)
Unirep_Test_data,PHY_Test_data=Trans_data(Test_X)
Train_Y=Train_Y.astype('float64')
print(Train_Y.shape)
print(PHY_Train_data.shape)
Valid_Y=Valid_Y.astype('float64')
Test_Y=Test_Y.astype('float64')
Y = keras.utils.to_categorical(Train_Y)
Y2 = keras.utils.to_categorical(Valid_Y,num_classes=4)
Y4 = keras.utils.to_categorical(Valid_Y1,num_classes=4)
Y3 = keras.utils.to_categorical(Test_Y)
import math
print(Counter(Train_Y))
# smo = SMOTE(sampling_strategy={2:4000,3:500 },random_state=42)
# PHY_Train_data = PHY_Train_data.astype('float64')
# PHY_Train_data_S, Train_Y_S = smo.fit_resample(PHY_Train_data, Train_Y)
# print(Counter(Train_Y_S))
# Y_S=keras.utils.to_categorical(Train_Y_S)
PHY_Train_data=np.reshape(PHY_Train_data,(-1,1,2,2))
# PHY_Train_data_S=np.reshape(PHY_Train_data_S,(-1,1,2,2))
PHY_Valid_data=np.reshape(PHY_Valid_data,(-1,1,2,2))
PHY_Test_data=np.reshape(PHY_Test_data,(-1,1,2,2))
# print(PHY_Train_data_S.shape)
# def get_class_weight(labels_dict):
#     """计算数据集不同类别的占比权重"""
#     total = sum(labels_dict.values())
#     max_num = max(labels_dict.values())
#     mu = 1.0 / (total / max_num)
#     class_weight = dict()
#     for key, value in labels_dict.items():
#         score = math.log(mu * total / float(value))
#         class_weight[key] = score if score > 1.0 else 1.0
#     return class_weight

# labels_dict = {0: 13774, 1: 3790, 2: 1320, 3: 435}  # 不平衡数据集
# Class_weight=get_class_weight(labels_dict)
# print(Class_weight)
# Unirep_Train Model
target_names = ['class 0 (Negtive)', 'class 1 (Endolysin)','class 2 (VAL)','class 3 (Holin)']
Model_dir="/share/lijianfeng/final/Model/MIX/Normal/tmp/"
if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)
HOME_DIR = Model_dir
os.chdir(Model_dir)
## paraments
numEpochs = 40
batchSize = 16
# LR = 0.00005
DROPOUT = 0.5
EPSILON = 0.2
DECAY = 0.001
DROP = 0.4
EPOCHS_DROP = 5
reg1 = regularizers.l1(0.0825)
reg2 = regularizers.l2(0.1750)
CLASS_WEIGHT1 = {0:0.8,1:4.8,2:10.37,3:12.79}
CLASS_WEIGHT2 = {0:1.4,1:6.1,2:9.37,3:8.79}
## Unirep Model
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))
def step_decay(epoch):
        initial_lrate = LR #初始学习率
        drop = DROP #学习率降低指数
        epochs_drop = EPOCHS_DROP #学习率降低周期
        lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
        return lrate
checkpointer1 = ModelCheckpoint(
    filepath='Unirep_best_weights.hdf5',
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=2
    )
checkpointer2 = ModelCheckpoint(
    filepath='PHY_best_weights.hdf5',
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=2
    )
checkpointer3 = ModelCheckpoint(
    filepath='MIX_best_weights.hdf5',
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=2
    )
earlyStoper = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    restore_best_weights=True, 
    # min_delta =0.75,
    patience = 5,
    verbose=2
    )
# learning schedule callback
loss_history1 = LossHistory()
loss_history2 = LossHistory()
loss_history3 = LossHistory()
lrate1 = LearningRateScheduler(step_decay)
lrate2 = LearningRateScheduler(step_decay)
lrate3 = LearningRateScheduler(step_decay)
callbacks_list1 = [loss_history1, lrate1, checkpointer1]
callbacks_list2 = [loss_history2, lrate2, checkpointer2]
callbacks_list3 = [loss_history3, lrate3, checkpointer3]
# define auc
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
#Keras implementation of Dice_BCE loss
X_input_1 = Input(shape=(19,25,4))

# ##
conv1 = Conv2D(
    filters=64, 
    kernel_size=10, 
    kernel_regularizer=reg2, 
    kernel_initializer='TruncatedNormal',
    activation='relu', 
    dilation_rate=(1, 1),
    strides=2,
    padding ='same', 
    use_bias=False)(X_input_1)
conv1 = BatchNormalization(axis=-1, center=True, scale=False, name = "embedding")(conv1)
conv1 = AveragePooling2D(pool_size = 2, strides = 2,padding = 'same')(conv1)
conv1 = Flatten()(conv1)
conv1 = Dense(units=64, activation='relu')(conv1)


conv2 = Conv2D(
    filters=64, 
    kernel_size=5, 
    kernel_regularizer=reg2, 
    kernel_initializer='TruncatedNormal',
    activation='relu', 
    dilation_rate=(2, 2),
    padding ='same', 
    use_bias=False)(X_input_1)
conv2 = BatchNormalization(axis=-1, center=True, scale=False)(conv2)
conv2 = AveragePooling2D(pool_size = 2, strides = 2,padding = 'same')(conv2)
conv2 = Flatten()(conv2)
conv2 = Dense(units=64, activation='relu')(conv2)

conv3 = Conv2D(
    filters=64, 

    kernel_size=1, 
    kernel_regularizer=reg2, 
    kernel_initializer='TruncatedNormal',
    activation='relu', 
    padding ='same', 
    use_bias=False)(X_input_1)
conv3 = BatchNormalization(axis=-1, center=True, scale=False)(conv3)
conv3 = AveragePooling2D(pool_size = 2, strides = 2,padding = 'same')(conv3)
conv3 = Flatten()(conv3)
conv3 = Dense(units=64, activation='relu')(conv3)
# c1 = K.concatenate([a, b], axis=0)
# c1 = K.concatenate([a, b], axis=0)
# c1 = K.concatenate([a, b], axis=0)
combined1 = add([conv1, conv2])
combined1 = add([combined1, conv3])
combined2 = multiply([conv1, conv2])
combined2 = multiply([combined2, conv3])
combined3 = average([combined1, combined2])
combined3 = Dense(128, activation='relu')(combined3)
combined3 = Dense(64, activation='relu')(combined3)

conv2 = Embedding(
    input_dim=1900, 
    output_dim=64, 
    embeddings_regularizer=reg1, 
    name='EXP_input')(X_input_1) 
conv2 = GlobalAvgPool3D()(conv2)
conv2 = Dense(units=64, activation='relu')(conv2)
combined1 = add([combined3, conv2])
combined2 = multiply([combined3, conv2])
combined = average([combined1, combined2])
combined = Dense(128, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined = Dense(64, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined_M = Dense(4, activation='softmax')(combined)
classifier1 = Model(inputs=X_input_1, outputs=combined_M)
print("Model summary: \n", classifier1.summary())
LR=0.0008
adam = keras.optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)
classifier1.compile(
    optimizer=adam, 
    loss='CategoricalCrossentropy',    
    metrics=['acc'])
training_start_time = timeit.default_timer()

history = classifier1.fit(
    x = Unirep_Train_data,
    y = Y,
    batch_size = batchSize,
    epochs = numEpochs,
    validation_data=(Unirep_Valid_data,Y2),
    shuffle=True,
    class_weight=CLASS_WEIGHT1,
    callbacks=callbacks_list1,
    verbose=2
    )
training_end_time1 = timeit.default_timer()
print("Unirep_Training time: {:10.2f} min. \n" .format((training_end_time1 - training_start_time) / 60))   
train_scores = classifier1.evaluate(Unirep_Train_data, Y, verbose = 0 )
test_scores = classifier1.evaluate(Unirep_Valid_data, Y2, verbose = 0 )
print("Unirep_Training accuracy: {:6.2f}%".format(train_scores[1]*100))
print("Unirep_Validation accuracy: {:6.2f}%".format(test_scores[1]*100))
model_json = classifier1.to_json()
with open("./Unirep_classifier.json", "w") as json_file:
    json_file.write(model_json)
json_file = open('./Unirep_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_Unirep = model_from_json(best_model_json)
# load best weights into new model
classifier1.save_weights("./Unirep_model_weights.hdf5")
best_model_Unirep.load_weights("./Unirep_best_weights.hdf5")
print("Loaded best Unirep Model weights from disk")
print(best_model_Unirep.input)
print(best_model_Unirep.layers[-2].output)
Unirep_h=Model(inputs=best_model_Unirep.input,outputs=best_model_Unirep.layers[-2].output)
Unirep_h_fea= Unirep_h.predict(Unirep_Train_data)
Unirep_h_fea=np.array(Unirep_h_fea)


############
X_input_2 = Input(shape=(1,2,2))
conv1 = Conv2D(
    filters=64, 
    kernel_size=2, 
    kernel_regularizer=reg2, 
    kernel_initializer='TruncatedNormal',
    activation='relu', 
    dilation_rate=(1, 1),
    strides=2,
    padding ='same', 
    use_bias=False)(X_input_2)
conv1 = BatchNormalization(axis=-1, center=True, scale=False)(conv1)
conv1 = AveragePooling2D(pool_size = 2, strides = 2,padding = 'same')(conv1)
conv1 = Flatten()(conv1)
conv1 = Dense(units=64, activation='relu')(conv1)
conv2=Flatten()(X_input_2)
conv2 = Dense(units=64, activation='relu')(conv2)
combined1 = add([conv1, conv2])
combined2 = multiply([conv1, conv2])
combined3 = average([combined1, combined2])
combined3 = Dense(128, activation='relu')(combined3)
combined3 = Dense(64, activation='relu')(combined3)
combined_P = Dense(4, activation='softmax')(combined3)
classifier2 = Model(inputs=X_input_2, outputs=combined_P)
print("Model summary: \n", classifier2.summary())

LR=0.00004

adam = keras.optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)
classifier2.compile(
    optimizer=adam, 
    loss='CategoricalCrossentropy',    
    metrics=['acc'])  
training_start_time = timeit.default_timer()
history = classifier2.fit(
    x = PHY_Train_data,
    y = Y,
    batch_size = batchSize,
    epochs = numEpochs,
    validation_data=(PHY_Valid_data,Y2),
    shuffle=True,
    class_weight=CLASS_WEIGHT2,
    callbacks=callbacks_list2,
    verbose=2
    )
training_end_time2 = timeit.default_timer()
print("PHY_Training time: {:10.2f} min. \n" .format((training_end_time2 - training_start_time) / 60))   
train_scores = classifier2.evaluate(PHY_Train_data, Y, verbose = 0 )
test_scores = classifier2.evaluate(PHY_Valid_data, Y2, verbose = 0 )
print("PHY_Training accuracy: {:6.2f}%".format(train_scores[1]*100))
print("PHY_Validation accuracy: {:6.2f}%".format(test_scores[1]*100))
model_json = classifier2.to_json()
target_names = ['class 0 (Negtive)', 'class 1 (Endolysin)','class 2 (VAL)','class 3 (Holin)']

with open("./PHY_classifier.json", "w") as json_file:
    json_file.write(model_json)
json_file = open('./PHY_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_PHY = model_from_json(best_model_json)
# # load best weights into new model
classifier2.save_weights("./PHY_model_weights.hdf5")
best_model_PHY.load_weights("./PHY_best_weights.hdf5")
print("Loaded best PHY Model weights from disk")
print(best_model_PHY.input)
print(best_model_PHY.layers[-2].output)
PHY_h=Model(inputs=best_model_PHY.input,outputs=best_model_PHY.layers[-2].output)
PHY_h_fea= PHY_h.predict(PHY_Train_data)
PHY_h_fea=np.array(PHY_h_fea)

######
print(Unirep_h_fea.shape)
print(PHY_h_fea.shape)
Total_Fea=np.c_[Unirep_h_fea,PHY_h_fea]
print(Total_Fea.shape)
Total_X=np.reshape(Total_Fea,(-1,4,4,8))
X_input_3 = Input(shape=(4,4,8))
conv1 = Conv2D(
    filters=64, 
    kernel_size=10, 
    kernel_regularizer=reg2, 
    kernel_initializer='TruncatedNormal',
    activation='relu', 
    dilation_rate=(1, 1),
    strides=2,
    padding ='same', 
    use_bias=False)(X_input_3)
conv1 = BatchNormalization(axis=-1, center=True, scale=False)(conv1)
conv1 = AveragePooling2D(pool_size = 3, strides = 3,padding = 'same')(conv1)
conv1 = BatchNormalization(axis=-1, center=True, scale=False)(conv1)
conv1 = AveragePooling2D(pool_size = 2, strides = 2,padding = 'same')(conv1)
conv1 = Flatten()(conv1)
conv1 = Dense(64, activation='relu')(conv1)
conv2=Flatten()(X_input_3)
conv2 = Dense(64, activation='relu')(conv2)
combined1= add([conv1, conv2])
combined2 = multiply([conv1, conv2])
combined= average([combined1, combined2])

combined = Dense(64, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined = Dense(16, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined_output = Dense(4, activation='softmax')(combined)
# combined = Dropout(rate=DROPOUT)(combined)
# combined_U = Dense(64, activation='relu')(combined)
# combined_P = Dense(64, activation='relu')(X_input_2)
# combined_M1 = add([combined_U, combined_P])
# combined_M2 = multiply([combined_U, combined_P])
# combined_M = average([combined_M1, combined_M2])
# combined_M = Dense(128, activation='relu')(combined_M)
# combined_M = Dropout(rate=DROPOUT)(combined_M)
# combined_M = Dense(64, activation='relu')(combined_M)
# combined_output = Dense(4, activation='softmax')(combined_M)
classifier3 = Model(inputs=X_input_3, outputs=combined_output)
print("Model summary: \n", classifier3.summary())
LR=0.001
adam = keras.optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)
classifier3.compile(
    optimizer=adam, 
    loss='CategoricalCrossentropy',     
    metrics=['acc'])
training_start_time = timeit.default_timer()
history = classifier3.fit(
    x = Total_X,
    y = Y,
    batch_size = batchSize,
    epochs = 80,
    shuffle=True,
    validation_split=0.1,
    class_weight=CLASS_WEIGHT1,
    callbacks=callbacks_list3,
    verbose=2
    )
training_end_time3 = timeit.default_timer()
print("Training time: {:10.2f} min. \n" .format((training_end_time3 - training_start_time) / 60))   
train_scores = classifier3.evaluate(Total_X, Y, verbose = 0 )
# test_scores = classifier3.evaluate(PHY_Valid_data, Y2, verbose = 0 )
print("Training accuracy: {:6.2f}%".format(train_scores[1]*100))
# print("Validation accuracy: {:6.2f}%".format(test_scores[1]*100))
model_json = classifier3.to_json()
with open("./Last_classifier.json", "w") as json_file:
    json_file.write(model_json)
json_file = open('./Last_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_MIX = model_from_json(best_model_json)
classifier3.save_weights("./MIX_model_weights.hdf5")
best_model_MIX.load_weights("./MIX_best_weights.hdf5")
print("Loaded best MIX Model weights from disk")



# serialize weights to HDF5

with open("./y_true",'a+') as f1:
    np.savetxt(f1,Y2,fmt="%s")
del f1
Unirep_h=Model(inputs=best_model_Unirep.input,outputs=best_model_Unirep.layers[-2].output)
Unirep_h_test_fea=Unirep_h.predict(Unirep_Test_data)
Unirep_h_test_fea=np.array(Unirep_h_test_fea)
PHY_h=Model(inputs=best_model_PHY.input,outputs=best_model_PHY.layers[-2].output)
PHY_h_test_fea=PHY_h.predict(PHY_Test_data)
PHY_h_test_fea=np.array(PHY_h_test_fea)
print(PHY_h_test_fea.shape)
print(Unirep_h_test_fea.shape)
Total_test_Fea=np.c_[Unirep_h_test_fea,PHY_h_test_fea]
Total_test_Fea=np.reshape(Total_test_Fea,(-1,4,4,8))
print(Total_test_Fea.shape)
test_Y_pred = best_model_MIX.predict(Total_test_Fea)

Unirep_h_train_fea=Unirep_h.predict(Unirep_Train_data)
Unirep_h_train_fea=np.array(Unirep_h_train_fea)
PHY_h_train_fea=PHY_h.predict(PHY_Train_data)
PHY_h_train_fea=np.array(PHY_h_train_fea)
Total_train_Fea=np.c_[Unirep_h_train_fea,PHY_h_train_fea]
Total_train_Fea=np.reshape(Total_train_Fea,(-1,4,4,8))
train_Y_pred = best_model_MIX.predict(Total_train_Fea)
U_test_Y_pred=best_model_Unirep.predict(Unirep_Test_data)
P_test_Y_pred=best_model_PHY.predict(PHY_Test_data)
U_test_y_pred=[]
P_test_y_pred=[]
true_y=[]
for item in Y3:
    true_y.append(np.argmax(item))
train_y=[]
for item in Y:
    train_y.append(np.argmax(item))
for item in U_test_Y_pred:
    U_test_y_pred.append(np.argmax(item))
for item in P_test_Y_pred:
    P_test_y_pred.append(np.argmax(item))
with open ("./Unirep.txt","a") as f1:
    print(classification_report(true_y, U_test_y_pred, target_names=target_names),file=f1)
with open ("./Phy.txt","a") as f1:
    print(classification_report(true_y, P_test_y_pred, target_names=target_names),file=f1)
    print(confusion_matrix(true_y, P_test_y_pred),file=f1)
test_y_pred = []
train_y_pred=[]
print(test_Y_pred.shape)
for item in test_Y_pred:
    test_y_pred.append(np.argmax(item))
for item in train_Y_pred:
    train_y_pred.append(np.argmax(item))

test_y_pred=np.array(test_y_pred)
train_y_pred=np.array(train_y_pred)
true_y=np.array(true_y)
print(true_y.shape)
print(test_y_pred.shape)
print('The f1 score on traindatasets for the model is:',f1_score(train_y, train_y_pred,average='weighted'))
print('The f1 score on validdatasets for the model is:',f1_score(true_y, test_y_pred,average='weighted'))
target_names = ['class 0 (Negtive)', 'class 1 (Endolysin)','class 2 (VAL)','class 3 (Holin)']
with open("./model_MIX.txt", "a+") as f3:
    print(classification_report(true_y, test_y_pred, target_names=target_names),file=f3)
    print('The f1 score on traindatasets for the model model is:',f1_score(train_y, train_y_pred,average='weighted'),file=f3)
    print('The f1 score on validdatasets for the model is:',f1_score(true_y, test_y_pred,average='weighted'),file=f3)
    print(confusion_matrix(true_y, test_y_pred),file=f3)
    print("\n",file=f3)
