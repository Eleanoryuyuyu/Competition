# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:09:29 2019

@author: node5
"""

import numpy as np
import pandas as pd






#需要删除的id编号
a = [1001,1010,1011,1024,1028,1088,1119,1122,1198,1245,1310,1350,1496,1606,1626, \
 1681,1701,1756,1758,1759,1761,1762,1767,1801,1810,1823,1829,1834,1842,1848, \
 1850,1852,1857,1861,1869,1874,1876,1877,1879,1882,1888,1889,1892,1897,1901, \
 1907,1915,1923,1932,1940,1945,1950,1952,1960,1967,1977,1979,1983,1984,2149, \
 2435,2502,2518,2582,2586,2630,2674,2677,2689,2703,2713,2728,2734,2876,3028, \
 3378,3544,3547,3605,3660,3669,3699,3778,3787,3790]
###############################################################################

#获取finance特征
feature_finance = pd.read_csv('../feature_finance.csv')
#获取target特征
feature_target = pd.read_csv('../feature_target.csv')
###############################################################################
#获取power特征
feature_power = pd.read_csv('../feature_power.csv')
b = feature_power['id'].values.tolist()
#找出可能需要删除的id编号
c = []
for i in b:
    if i in a:
        c.append(i)

for j in c:
    index_1 = feature_power.loc[feature_power['id']==j].index.tolist()
    feature_power.drop(index_1[0], inplace=True)
    #feature_power = feature_power.reset_index(drop=True)
feature_newpower = feature_power.reset_index(drop=True)
###############################################################################
#获取house_year特征
feature_house_year = pd.read_csv('../feature_house_year.csv')
d = feature_house_year['id'].values.tolist()
#找出可能需要删除的id编号
e = []
for i in d:
    if i in a:
        e.append(i)

for j in e:
    index_2 = feature_house_year.loc[feature_house_year['id']==j].index.tolist()
    feature_house_year.drop(index_2[0], inplace=True)
    #feature_power = feature_power.reset_index(drop=True)
feature_newhouse_year = feature_house_year.reset_index(drop=True)
###############################################################################
#获取标签
label = pd.read_excel('../企业评分.xlsx')
label.rename(columns={'企业编号':'id'}, inplace=True)
###############################################################################

data = pd.merge(feature_finance, feature_target, on='id')
data = pd.merge(data, feature_newpower, how='outer', on='id')
data = pd.merge(data, feature_newhouse_year, how='outer', on='id')
data = pd.merge(data, label, on='id')

data.fillna(0, inplace=True)
data.to_csv('../feature_combine.csv', index=False)









