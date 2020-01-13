# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:09:43 2019

@author: Administrator
"""

import pandas as pd
from keras import models
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np


###############################################################################
#归一化
scaler_1 = MinMaxScaler()
scaler_2 = MinMaxScaler()
#获取数据
data = pd.read_csv('../feature_combine.csv')
data = data.loc[:, 1:-1].values
label = data.loc[:, '企业总评分'].values
#进行归一化
newdata = scaler_1.fit_transform(data)
newlabel = scaler_2.fit_transform(label.reshape(-1, 1))
#拆分成训练和测试集
train = newdata[:2750]
test = newdata[2750:]
train_label = newlabel[:2750]
test_label = newlabel[2750:]

###############################################################################
#构建神经网络模型
def build_model():
    #初始化
    model = models.Sequential()
    #搭建
    model.add(Dense(100, activation='relu',input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    #编译网络
    model.compile(optimizer='adam', loss='mse')
    return model


###############################################################################
n_fold = 5
kf = KFold(n=train.shape[0], n_folds=n_fold, shuffle=True, random_state=2019)

n = 0
for index_train, index_eval in kf:

    x_train, x_eval = train[index_train], train[index_eval]
    y_train, y_eval = train_label[index_train], train_label[index_eval]

    model = build_model()
    model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=0)
    testpreds = model.predict(test)

    if n > 0:
        totalpreds = totalpreds + testpreds
    else:
        totalpreds = testpreds
    
    n += 1
###############################################################################
totalpreds = totalpreds / n
totalpreds = scaler_2.inverse_transform(totalpreds)
test_label = scaler_2.inverse_transform(test_label)
lin_mse = mean_squared_error(test_label, totalpreds)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)















