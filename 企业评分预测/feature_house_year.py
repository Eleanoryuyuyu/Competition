# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:31:42 2019

@author: Administrator
"""

import numpy as np
import pandas as pd


###############################################################################
#经营状态量化
def info_1(x):
    a = {'开业':1, '正常开业':1, '开业/正常经营':1, '存续':1, '其他':0.8, '歇业':0.1, \
          '清算':0.1, '停业':0.5}
    return a[x]

#企业信息量化    
def info_2(x):
    b = {'是':1.25, '有':1.25, '否':1, '无':1}
    return b[x]

#企业信息量化    
def info_3(x):
    b = {'是':0.75, '有':0.75, '否':1, '无':1}
    return b[x]

def collect(x):
    if '人' in x:
        return float(x.strip('人')) 
    else:
        return float(x)
###############################################################################



def get_housedata(filename_1):
    
    #地块公示表
    earthed = pd.read_excel(filename_1)
    earth_columns = ['企业编号', '土地面积（公顷）']
    earth = earthed[earth_columns].copy(deep=True)
                             
    #生成统计特征
    earth_stat = earth.groupby('企业编号')['土地面积（公顷）'].agg({'earth'+'_mean':'mean', \
                                              'earth'+'_sum':'sum'})
    earth_stat = earth_stat.reset_index()
                    
    earth_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################
    
    return earth_stat


def get_yeardata(filename_2, filename_3, filename_4, filename_5):

    #网站或网点信息表
    webed = pd.read_excel(filename_2)
    web_columns = ['企业编号', '年报年份']
    web = webed[web_columns]
    
    #填充缺失值
    web.fillna({'年报年份':0}, inplace=True)
                        
    #生成统计特征
    web_stat = web.groupby('企业编号')['年报年份'].count()
    web_stat = web_stat.reset_index()
                    
    web_stat.rename(columns={'企业编号':'id', '年报年份':'web_count'}, inplace=True)
    ###########################################################################

    #对外投资信息表
    ofdied = pd.read_excel(filename_3)
    ofdi_columns = ['企业编号', '投资金额']
    ofdi = ofdied[ofdi_columns]
    
    #填充缺失值
    ofdi.fillna({'投资金额':0}, inplace=True)
                             
    #生成统计特征
    ofdi_stat = ofdi.groupby('企业编号')['投资金额'].agg({'ofdi'+'_mean':'mean', 'ofdi'+'_sum':'sum'})
    ofdi_stat = ofdi_stat.reset_index()
                    
    ofdi_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #企业基本信息表
    basised = pd.read_excel(filename_4)
    basis_columns = ['企业编号', '企业经营状态', '是否有网站或网点', '企业是否有投资信息或购买其他公司股权', \
                    '有限责任公司本年度是否发生股东股权转', '是否提供对外担保']
    basis = basised[basis_columns].copy(deep=True)
    
    #填充缺失值
    for basis_col in basis_columns[2:]:
        basis.fillna({basis_col:'否'}, inplace=True)
    
    #删除经营列缺失的行
    basis = basis.dropna(axis=0, how='any')
    basisddt = basis.reset_index(drop=True)
    
    #将字符串量化
    basisddt['企业经营状态'] = basisddt['企业经营状态'].apply(lambda x: info_1(x))
    for basis_col in basis_columns[2:4]:
        basisddt[basis_col] = basisddt[basis_col].apply(lambda x: info_2(x))
    
    for basis_col in basis_columns[4:]:
        basisddt[basis_col] = basisddt[basis_col].apply(lambda x: info_3(x))
    
    #引入特征
    basisddt['actual_basis'] = basisddt['企业经营状态'] * basisddt['是否有网站或网点'] * \
                               basisddt['企业是否有投资信息或购买其他公司股权'] * \
                               basisddt['有限责任公司本年度是否发生股东股权转'] * \
                               basisddt['是否提供对外担保']
    
                             
    #生成统计特征
    basis_stat = basisddt.groupby('企业编号')['actual_basis'].agg({'actbasis'+'_mean':'mean', \
                         'actbasis'+'_max':'max', 'actbasis'+'_min':'min', 'actbasis'+'_sum':'sum'})
    basis_stat = basis_stat.reset_index()
                    
    basis_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #社保信息表
    socaled = pd.read_excel(filename_5)
    socal_columns = ['企业编号', '城镇职工基本养老保险人数', '失业保险人数', '职工基本医疗保险人数', \
                    '工伤保险人数', '生育保险人数']
    socal = socaled[socal_columns].copy(deep=True)
    
    #填充缺失值
    for socal_col in socal_columns[1:]:
        socal.fillna({socal_col:'0'}, inplace=True)
    
    #提取数据    
    for socal_col in socal_columns[1:]:
        socal[socal_col] = socal[socal_col].apply(lambda x: collect(x))
        
    #引入特征
    socal['total_person'] = socal['城镇职工基本养老保险人数'] + socal['失业保险人数'] + \
                               socal['职工基本医疗保险人数'] + socal['工伤保险人数'] + \
                               socal['生育保险人数']
    
                             
    #生成统计特征
    socal_stat = socal.groupby('企业编号')['total_person'].agg({'totperson'+'_max':'max', 'totperson'+'_sum':'sum'})
    socal_stat = socal_stat.reset_index()
                    
    socal_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    house_year_feature = pd.merge(web_stat, ofdi_stat, how='outer', on='id')
    house_year_feature = pd.merge(house_year_feature, basis_stat, how='outer', on='id')
    house_year_feature = pd.merge(house_year_feature, socal_stat, how='outer', on='id')
    house_year_feature.fillna(0, inplace=True)
    
    return house_year_feature









if __name__ == "__main__":
    filename_1='../购地-地块公示.xlsx'
    filename_2='../年报-网站或网点信息.xlsx'
    filename_3='../年报-对外投资信息.xlsx'
    filename_4='../年报-企业基本信息.xlsx'
    filename_5='../年报-社保信息.xlsx'
    house_ddt = get_housedata(filename_1)
    year_ddt = get_yeardata(filename_2, filename_3, filename_4, filename_5)
    house_year_ddt = pd.merge(house_ddt, year_ddt, how='outer', on='id')
    house_year_ddt.fillna(0, inplace=True)
    house_year_ddt.to_csv('../feature_house_year.csv', index=False)






















    