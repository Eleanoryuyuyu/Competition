# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:42:51 2019

@author: Administrator
"""


import numpy as np
import pandas as pd


#将专利量化
def func(x):
    if '发明专利' in x:
        return 12
    elif '发明授权' in x:
        return 6
    elif '发明授权更正' in x:
        return 5
    elif '发明公布' in x:
        return 3
    elif '发明公布更正' in x:
        return 2
    elif '实用新型' in x:
        return 6
    elif '实用新型更正' in x:
        return 3
    elif '外观设计' in x:
        return 5
    elif '外观设计更正' in x:
        return 2.5
    else:
        return 0

#招投标量化
def trans(x):
    if '其它' in x or '其它' in x:
        return 0
    elif '招标' in x or '单一' in x or '废标' in x or '流标' in x or '违规' in x:
        return 1
    elif '预告' in x or '询价' in x:
        return 2
    elif '竞谈' in x or '公开招标' in x or '竞争性谈判' in x or '竞价' in x:
        return 3
    elif '中标' in x:
        return 4
    elif '变更' in x or '结果变更' in x:
        return 5
    elif '合同' in x:
        return 6
    elif '成交' in x:
        return 7
    elif '拟建' in x:
        return 8
    elif '验收' in x:
        return 9
    else:
        return int(x)

#状态量化       
def vall(x):
    a = ['已注销', '注销', '已撤销', '过期失效', '撤销', '暂停', '已暂停', '当前批件', \
         '已过期', '旧版', '无效（依申请注销）', '无效（逾期未换证）', '证书注销']
    b = ['有效', '延续', '正常', '历史批件', '注销(非申请)', '新立', '变更']
    if x in a:
        return 0
    if x in b:
        return 1    
        
#量化修正
def cor_replace(x, index, y):
    for i in index:
        x.loc[i] = y.loc[i]
    return x

#著作权量化    
def num_count(x):
    if x == '0':
        return 0
    else:
        return 1

#产品量化       
def inform(x):
    a = ['android', 'ios']
    b = ['wechat', 'weibo']
    c = ['website', 'miniapp']
    if x in a:
        return 3
    if x in b:
        return 2
    if x in c:
        return 1
        
#信用量化
def honesty(x):
    a = {'一般经济区域':1, '经济技术开发全区':2, '经济技术开发区':2, '高新技术产业开发区':2, \
         '保税物流园区':2, '保税区':2, '保税港区、综合保税区':3, '经济特区':4}
    return a[x]

#海关注销企业信用更新
def replace(x, index):
    for i in index:
        x.loc[i] = 0
    return x               
###############################################################################

def get_powerdata(filename_1, filename_2, filename_3, filename_4, filename_5, filename_6):
    
    #专利表
    patented = pd.read_excel(filename_1)
    patent_columns = ['企业编号', '专利类型']
    patent = patented[patent_columns].copy(deep=True)

    #专利量化
    patent['专利类型'] = patent['专利类型'].apply(lambda x: func(x))
    
    #生成统计特征
    patent_stat = patent.groupby('企业编号')['专利类型'].agg({'patent'+'_mean':'mean', 'patent'+'_sum':'sum'})
    patent_stat = patent_stat.reset_index()
    
    patent_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #招投标表
    tendered = pd.read_excel(filename_2)
    tender_columns = ['企业编号', '公告类型', '中标或招标']
    tender = tendered[tender_columns].copy(deep=True)

    #填充缺失值
    tender.fillna({'公告类型':'0'}, inplace=True)
    
    #招投标量化
    for tender_col in tender_columns[1:]:
        tender[tender_col] = tender[tender_col].apply(lambda x: trans(x))
    
    #量化修正
    index_1 = tender.loc[tender['公告类型']==0].index.tolist()
    tender['公告类型'] = cor_replace(tender['公告类型'], index_1, tender['中标或招标'])
    
    tenderddt = tender[['企业编号', '公告类型']]
    #生成统计特征
    tenderddt_stat = tenderddt.groupby('企业编号')['公告类型'].agg({'tender'+'_mean':'mean', 'tender'+'_sum':'sum'})
    tenderddt_stat = tenderddt_stat.reset_index()
    
    tenderddt_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #债券信息表
    bonded = pd.read_excel(filename_3)
    bond_columns = ['企业编号', '计划发行总额（亿元）', '票面利率（%）']
    bond = bonded[bond_columns].copy(deep=True)
    
    #删除缺失值的行
    bond = bond.dropna(axis=0, how='any')
    bond_ddt = bond.reset_index(drop=True)
    
    #生成利息特征
    bond_ddt['利息'] = bond_ddt['计划发行总额（亿元）'] * bond_ddt['票面利率（%）'] / 100
    
    #生成统计特征
    bond_ddt_1 = bond_ddt.groupby('企业编号')['计划发行总额（亿元）'].agg({'plan'+'_mean':'mean', 'plan'+'_sum':'sum'})
    bond_ddt_1 = bond_ddt_1.reset_index()
    bond_ddt_2 = bond_ddt.groupby('企业编号')['利息'].agg({'inter'+'_mean':'mean', 'inter'+'_sum':'sum'})
    bond_ddt_2 = bond_ddt_2.reset_index()
    bond_ddt_3 = bond_ddt.groupby('企业编号')['票面利率（%）'].agg({'interate'+'_mean':'mean'})
    bond_ddt_3 = bond_ddt_3.reset_index()
    bond_data = pd.merge(bond_ddt_1, bond_ddt_2, on='企业编号')
    bond_data = pd.merge(bond_data, bond_ddt_3, on='企业编号')
      
    bond_data.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #资质认证表
    aptied = pd.read_excel(filename_4)
    apti_columns = ['企业编号', '状态']
    apti = aptied[apti_columns].copy(deep=True)
        
    #删除缺失值的行
    apti = apti.dropna(axis=0, how='any')
    apti_ddt = apti.reset_index(drop=True)
    
    #状态量化
    apti_ddt['状态'] = apti_ddt['状态'].apply(lambda x: vall(x))
    
    #生成统计特征
    apti_ddt_stat = apti_ddt.groupby('企业编号')['状态'].agg({'aptitude'+'_mean':'mean', 'aptitude'+'_sum':'sum'})
    apti_ddt_stat = apti_ddt_stat.reset_index()
        
    apti_ddt_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #作品著作权表
    worked = pd.read_excel(filename_5)
    work_columns = ['企业编号', '作品著作权登记日期']
    work = worked[work_columns].copy(deep=True)
            
    #填充缺失值
    work.fillna({'作品著作权登记日期':'0'}, inplace=True)
        
    #著作权量化
    work['作品著作权登记日期'] = work['作品著作权登记日期'].apply(lambda x: num_count(x))
        
    #生成统计特征
    work_stat = work.groupby('企业编号')['作品著作权登记日期'].agg({'work'+'_mean':'mean', 'work'+'_sum':'sum'})
    work_stat = work_stat.reset_index()
            
    work_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #软件著作权表
    softed = pd.read_excel(filename_6)
    soft_columns = ['企业编号', '软件全称']
    soft = softed[soft_columns].copy(deep=True)
                           
    #生成统计特征
    soft_stat = soft.groupby('企业编号')['软件全称'].count()
    soft_stat = soft_stat.reset_index()
            
    soft_stat.rename(columns={'企业编号':'id', '软件全称':'soft_count'}, inplace=True)
    ###########################################################################

    power_feature = pd.merge(patent_stat, tenderddt_stat, how='outer', on='id')
    power_feature = pd.merge(power_feature, bond_data, how='outer', on='id')
    power_feature = pd.merge(power_feature, apti_ddt_stat, how='outer', on='id')
    power_feature = pd.merge(power_feature, work_stat, how='outer', on='id')
    power_feature = pd.merge(power_feature, soft_stat, how='outer', on='id')
    power_feature.fillna(0, inplace=True)
    
    return power_feature

    

def get_productdata(filename_7, filename_8, filename_9):

    #产品表
    gained = pd.read_excel(filename_7)
    gain_columns = ['企业编号', '产品类型']
    gain = gained[gain_columns].copy(deep=True)
                               
    #量化
    gain['产品类型'] = gain['产品类型'].apply(lambda x: inform(x))
    
    #生成统计特征
    gain_stat = gain.groupby('企业编号')['产品类型'].agg({'gain'+'_mean':'mean', 'gain'+'_sum':'sum'})
    gain_stat = gain_stat.reset_index()
                
    gain_stat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #纳税表
    payed = pd.read_excel(filename_8)
    pay_columns = ['企业编号', '纳税A级年份']
    pay = payed[pay_columns].copy(deep=True)
                               
    #生成统计特征
    pay_stat = pay.groupby('企业编号')['纳税A级年份'].count()
    pay_stat = pay_stat.reset_index()
                
    pay_stat.rename(columns={'企业编号':'id', '纳税A级年份':'pay_count'}, inplace=True)
    ###########################################################################

    #海关进出口信用表
    honested = pd.read_excel(filename_9)
    honest_columns = ['企业编号', '经济区划', '海关注销标志']
    honest = honested[honest_columns].copy(deep=True)
    
    #填充缺失值
    honest.fillna({'经济区划':'一般经济区域'}, inplace=True)
    honest.fillna({'海关注销标志':'正常'}, inplace=True)                           
    
    #信用等级量化
    honest['经济区划'] = honest['经济区划'].apply(lambda x: honesty(x))
    
    #信用更新
    index = honest.loc[honest['海关注销标志']=='注销'].index.tolist()
    honest['经济区划'] = replace(honest['经济区划'], index)
    
    honest_stat = honest[['企业编号', '经济区划']].copy(deep=True)            
    honest_stat.rename(columns={'企业编号':'id', '经济区划':'honest_level'}, inplace=True)
    ###########################################################################

    product_feature = pd.merge(gain_stat, pay_stat, how='outer', on='id')
    product_feature = pd.merge(product_feature, honest_stat, how='outer', on='id')
    product_feature.fillna(0, inplace=True)
    
    return product_feature


  
    
    
if __name__ == "__main__":
    filename_1='../专利.xlsx'
    filename_2='../招投标.xlsx'
    filename_3='../债券信息.xlsx'
    filename_4='../资质认证.xlsx'
    filename_5='../作品著作权.xlsx'
    filename_6='../软件著作权.xlsx'
    filename_7='../产品.xlsx'
    filename_8='../纳税A级年份.xlsx'
    filename_9='../海关进出口信用.xlsx'
    power_ddt = get_powerdata(filename_1, filename_2, filename_3, filename_4, filename_5, filename_6)
    product_ddt = get_productdata(filename_7, filename_8, filename_9)
    power_ddt = pd.merge(power_ddt, product_ddt, how='outer', on='id')
    power_ddt.fillna(0, inplace=True)
    power_ddt.to_csv('../feature_power.csv', index=False)















