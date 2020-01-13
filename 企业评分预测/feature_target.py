# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:02:31 2019

@author: Administrator
"""

import numpy as np
import pandas as pd


###############################################################################
#文本转化成浮点数
def func(x):
    if '万亿' in x:
	    return float(x.strip('万亿')) * 10 ** 12
    elif '亿' in x:
        return float(x.split('亿')[0]) * 10 ** 8
    elif '万' in x:
        return float(x.split('万')[0]) * 10 ** 4
    else:
        return float(x)

#去掉字符串中的逗号
def string_delete(x):
    if ',' in x:
        return x.replace(',', '')
    else:
        return x

#将-1的值替换为平均值
def replace(x, index, mean):
    for i in index:
        x.loc[i] = mean.loc[i]
    return x
###############################################################################

def get_target(filename_1, filename_2, filename_3, filename_4, filename_5):
    #每股指标表
    per_shared = pd.read_excel(filename_1)
    per_columns = ['企业编号', '日期' ,'基本每股收益(元)', '每股净资产(元)', \
                   '每股公积金(元)', '每股未分配利润(元)', '每股经营现金流(元)']
    per_share = per_shared[per_columns].copy(deep=True)

    #更换列名
    for per_index, per_col in enumerate(per_columns[2:]):
        per_share.rename(columns={per_col:'per_' + str(per_index+1)}, inplace=True)
    per_newcolumns = per_share.columns.values.tolist()
    
    #删除缺失值过多的行
    per_share = per_share.dropna(axis=0, how='any')
    per_shareddt = per_share.reset_index(drop=True)
    
    #将--替换为0
    for per_col in per_newcolumns[2:]:
        per_shareddt.loc[per_shareddt[per_col]=='--', per_col] = float(0)
        per_shareddt[per_col] = per_shareddt[per_col].astype(float)
    
    #生成统计特征
    per_count = per_shareddt.groupby('企业编号')[per_newcolumns[2]].count()
    per_count = per_count.reset_index()
    per_count.rename(columns={per_newcolumns[2]:'per_count'}, inplace=True)
    for per_col in per_newcolumns[2:]:
        per_stat = per_shareddt.groupby('企业编号')[per_col].agg({per_col+'_mean':'mean', \
                   per_col+'_max':'max', per_col+'_min':'min', \
                   per_col+'_std':'std'})
        per_stat = per_stat.reset_index()
        per_count = pd.merge(per_count, per_stat, on='企业编号')
    
    #将0替换为平均值
    #for perr_col in per_newcolumns[2:]:
        #index = per_shareddt.loc[per_shareddt[perr_col]==0].index.tolist()
        #per_shareddt[perr_col] = replace(per_shareddt[perr_col], index, \
                                 #per_shareddt[perr_col+'_mean'])

    per_count.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #财务风险指标表
    finan_risked = pd.read_excel(filename_2)
    risk_columns = ['企业编号', '资产负债率(%)', '流动负债/总负债(%)']
    finan_risk = finan_risked[risk_columns].copy(deep=True)
    
    #将--%替换为0.001%
    for risk_col in risk_columns[1:]:
        finan_risk.loc[finan_risk[risk_col]=='--%', risk_col] = '0.001%'

    #将字符串转换为浮点数
    for risk_col in risk_columns[1:]:
        finan_risk[risk_col] = finan_risk[risk_col].str.strip('%').astype(float)/100

    #更换列名
    for risk_index, risk_col in enumerate(risk_columns[1:]):
        finan_risk.rename(columns={risk_col:'risk_' + str(risk_index+1)}, inplace=True)
    risk_newcolumns = finan_risk.columns.values.tolist()
    
    #生成统计特征
    risk_count = finan_risk.groupby('企业编号')[risk_newcolumns[1]].count()
    risk_count = risk_count.reset_index()
    risk_count.rename(columns={risk_newcolumns[1]:'risk_count'}, inplace=True)
    for risk_col in risk_newcolumns[1:]:
        finan_stat = finan_risk.groupby('企业编号')[risk_col].agg({risk_col+'_mean':'mean', \
                   risk_col+'_max':'max', risk_col+'_min':'min', \
                   risk_col+'_std':'std'})
        finan_stat = finan_stat.reset_index()
        risk_count = pd.merge(risk_count, finan_stat, on='企业编号')
    
    #将0.001%替换为平均值
    #for riskk_col in risk_newcolumns[1:]:
        #index = finan_risk.loc[finan_risk[riskk_col]==0.00001].index.tolist()
        #finan_risk[riskk_col] = replace(finan_risk[riskk_col], index, \
                                #finan_risk[riskk_col+'_mean'])

    #更改列名
    #finan_risk['time'] = finan_risked['日期']
    #risk_count.drop(['risk_count'], axis=1, inplace=True)
    risk_count.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #成长能力指标表
    grow_ablited = pd.read_excel(filename_3)
    grow_columns = ['企业编号', '日期', '营业总收入(元)', '扣非净利润(元)', \
                    '营业总收入同比增长(元)', '扣非净利润同比增长(元)', \
                    '营业总收入滚动环比增长(元)', '扣非净利润滚动环比增长(元)']
    grow_ablit = grow_ablited[grow_columns].copy(deep=True)

    #将--替换为0//--%替换为0.001%
    for grow_col in grow_columns[2:4]:
        grow_ablit.loc[grow_ablit[grow_col]=='--', grow_col] = str(0)
    for grow_col in grow_columns[4:]:
        grow_ablit.loc[grow_ablit[grow_col]=='--%', grow_col] = '0.001%'
    
    #更换列名
    for grow_index, grow_col in enumerate(grow_columns[2:]):
        grow_ablit.rename(columns={grow_col:'grow_' + str(grow_index+1)}, inplace=True)
    grow_newcolumns = grow_ablit.columns.values.tolist()
    
    #将文本替换为浮点数
    for grow_col in grow_newcolumns[2:4]:
        grow_ablit[grow_col] = grow_ablit[grow_col].apply(lambda x: func(x))
    
    #将字符串替换为浮点数
    for grow_col in grow_newcolumns[4:]:
        grow_ablit[grow_col] = grow_ablit[grow_col].apply(lambda x: string_delete(x))
        grow_ablit[grow_col] = grow_ablit[grow_col].str.strip('%').astype(float)/100

   
    #生成统计特征
    grow_count = grow_ablit.groupby('企业编号')[grow_newcolumns[2]].count()
    grow_count = grow_count.reset_index()
    grow_count.rename(columns={grow_newcolumns[2]:'grow_count'}, inplace=True)  
    for grow_col in grow_newcolumns[2:]:
        grow_stat = grow_ablit.groupby('企业编号')[grow_col].agg({grow_col+'_mean':'mean', \
                   grow_col+'_max':'max', grow_col+'_min':'min', \
                   grow_col+'_std':'std'})
        grow_stat = grow_stat.reset_index()
        grow_count = pd.merge(grow_count, grow_stat, on='企业编号')
    
    #将0替换为平均值
    #for groww_col in grow_newcolumns[2:4]:
        #index = grow_ablit.loc[grow_ablit[groww_col]==0].index.tolist()
        #grow_ablit[groww_col] = replace(grow_ablit[groww_col], index, \
                                #grow_ablit[groww_col+'_mean'])

    #更改列名
    grow_count.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #盈利能力指标表
    prof_ablited = pd.read_excel(filename_4)
    prof_columns = ['企业编号', '日期', '加权净资产收益率(%)', '摊薄净资产收益率(%)', \
                    '净利率(%)', '实际税率(%)', '毛利率(%)']
    prof_ablit = prof_ablited[prof_columns].copy(deep=True)
    
    #删除毛利率列为--%的行
    delete_index = prof_ablit.loc[prof_ablit['毛利率(%)']=='--%'].index.tolist()
    prof_ablit = prof_ablit.drop(delete_index, axis=0)
    prof_ablitddt = prof_ablit.reset_index(drop=True)
    
    #将--%替换为0.001%
    for prof_col in prof_columns[2:]:
        prof_ablitddt.loc[prof_ablitddt[prof_col]=='--%', prof_col] = '0.001%'
    
    #将字符串替换为浮点数
    for prof_col in prof_columns[2:]:
        prof_ablitddt[prof_col] = prof_ablitddt[prof_col].str.strip('%').astype(float)/100

    #更换列名
    for prof_index, prof_col in enumerate(prof_columns[2:]):
        prof_ablitddt.rename(columns={prof_col:'prof_' + str(prof_index+1)}, inplace=True)
    prof_newcolumns = prof_ablitddt.columns.values.tolist()

    #生成统计特征
    prof_count = prof_ablitddt.groupby('企业编号')[prof_newcolumns[2]].count()
    prof_count = prof_count.reset_index()
    prof_count.rename(columns={prof_newcolumns[2]:'prof_count'}, inplace=True)
    for prof_col in prof_newcolumns[2:]:
        prof_stat = prof_ablitddt.groupby('企业编号')[prof_col].agg({prof_col+'_mean':'mean', \
                   prof_col+'_max':'max', prof_col+'_min':'min', \
                   prof_col+'_std':'std'})
        prof_stat = prof_stat.reset_index()
        prof_count = pd.merge(prof_count, prof_stat, on='企业编号')
    
    #将0.001%替换为平均值
    #for proff_col in prof_newcolumns[2:]:
        #index = prof_ablitddt.loc[prof_ablitddt[proff_col]==0].index.tolist()
        #prof_ablitddt[proff_col] = replace(prof_ablitddt[proff_col], index, \
                                #prof_ablitddt[proff_col+'_mean'])
    
    #更改列名
    prof_count.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #运营能力指标表
    oper_ablited = pd.read_excel(filename_5)
    oper_columns = ['企业编号', '日期', '总资产周转率(次)', '应收账款周转天数(天)', \
                    '存货周转天数(天)']
    oper_ablit = oper_ablited[oper_columns].copy(deep=True)
    
    #填充缺失值为0.001
    for oper_col in oper_columns[2:]:
        oper_ablit.fillna({oper_col:'0.001'}, inplace=True)
    
    #将--替换为0.001
    for oper_col in oper_columns[2:]:
        oper_ablit.loc[oper_ablit[oper_col]=='--', oper_col] = '0.001'
        oper_ablit[oper_col] = oper_ablit[oper_col].astype(float)
    
    #更换列名
    for oper_index, oper_col in enumerate(oper_columns[2:]):
        oper_ablit.rename(columns={oper_col:'oper_' + str(oper_index+1)}, inplace=True)
    oper_newcolumns = oper_ablit.columns.values.tolist()
    
    #生成统计特征
    oper_count = oper_ablit.groupby('企业编号')[oper_newcolumns[2]].count()
    oper_count = oper_count.reset_index()
    oper_count.rename(columns={oper_newcolumns[2]:'oper_count'}, inplace=True)
    for oper_col in oper_newcolumns[2:]:
        oper_stat = oper_ablit.groupby('企业编号')[oper_col].agg({oper_col+'_mean':'mean', \
                   oper_col+'_max':'max', oper_col+'_min':'min', \
                   oper_col+'_std':'std'})
        oper_stat = oper_stat.reset_index()
        oper_count = pd.merge(oper_count, oper_stat, on='企业编号')
    
    #将0.001替换为平均值
    #for operr_col in oper_newcolumns[2:]:
        #index = oper_ablit.loc[oper_ablit[operr_col]==0.001].index.tolist()
        #oper_ablit[operr_col] = replace(oper_ablit[operr_col], index, \
                                #oper_ablit[operr_col+'_mean'])
    
    #更换列名
    oper_count.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    target_feature = pd.merge(per_count, risk_count, on='id')
    target_feature = pd.merge(target_feature, grow_count, on='id')
    target_feature = pd.merge(target_feature, prof_count, on='id')
    target_feature = pd.merge(target_feature, oper_count, on='id')
    
    return target_feature


    
    
    
#filename_1每股指标表路径
#filename_2财务风险指标表路径
#filename_3成长能力指标表路径
#filename_4盈利能力指标表路径
#filename_5运营能力指标表路径
if __name__ == "__main__":
    filename_1='../每股指标表.xlsx'
    filename_2='../财务风险指标表.xlsx'
    filename_3='../成长能力指标表.xlsx'
    filename_4='../盈利能力指标表.xlsx'
    filename_5='../运营能力指标表.xlsx'
    target_ddt = get_target(filename_1, filename_2, filename_3, filename_4, filename_5)
    target_ddt.to_csv('../feature_target.csv', index=False)





