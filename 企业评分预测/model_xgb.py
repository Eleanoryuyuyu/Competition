# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:00:44 2019

@author: Administrator
"""



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

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

    model = xgb.XGBRegressor(objective='reg:linear', n_estimators=20000, min_child_weight=1, \
                             num_leaves=32, learning_rate=0.01, max_depth=5, n_jobs=20, \
                             subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)


    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_eval, y_eval)], \
                  early_stopping_rounds=500, verbose=2)
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












