# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:44:37 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.model_selection import KFold
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
import catboost as cb

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

n_fold = 5
kf = KFold(n=train.shape[0], n_folds=n_fold, shuffle=True, random_state=2019)

n = 0
for index_train, index_eval in kf:

    x_train, x_eval = train[index_train], train[index_eval]
    y_train, y_eval = train_label[index_train], train_label[index_eval]

    model = cb.CatBoostRegressor(objective='reg:linear', iterations=128, depth=5, \
                                 learning_rate=0.01, n_estimators=20000, l2_leaf_reg=5, \
                                 subsample=0.6, colsample_bylevel=1, max_bin=100, \
                                  min_samples_in_leaf=10, early_stopping_rounds=500, \
                                  loss_function='RMSE', random_state=2019)


    model.fit(x_train, y_train)
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
















