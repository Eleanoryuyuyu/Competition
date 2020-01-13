# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:21:12 2019

@author: Administrator
"""

import numpy as np
import pandas as pd


###############################################################################
#文本转化成浮点数
def fun(x):
    if '亿' in x:
        return float(x.split('亿')[0]) * 10 ** 8
    elif '万' in x:
        return float(x.split('万')[0]) * 10 ** 4
    else:
        return float(x)

#将-1的值替换为平均值
def replace(x, index, mean):
    for i in index:
        x.loc[i] = mean.loc[i]
    return x
###############################################################################

def get_finance(filename_1, filename_2, filename_3, filename_4):
    
    #现金流量表
    cashed = pd.read_excel(filename_1)
    cash_columns = ['企业编号', '经营:经营活动产生的现金流量净额(元)', \
    '投资:投资活动产生的现金流量净额(元)', '筹资活动产生的现金流量净额(元)']
    cash = cashed[cash_columns].copy(deep=True)
    
    cash.fillna({'投资:投资活动产生的现金流量净额(元)':'-1'}, inplace=True)
    cash.fillna({'筹资活动产生的现金流量净额(元)':'-1'}, inplace=True)
    
    for col in cash_columns[2:]:
        cash[col] = cash[col].apply(lambda x: str(-1) if x=='--' else x)
    
    for col in cash_columns[1:]:
        cash[col] = cash[col].apply(lambda x: fun(x))
    
    #引入特征
    cash['total_cash'] = cash['经营:经营活动产生的现金流量净额(元)'] + \
                         cash['投资:投资活动产生的现金流量净额(元)'] + \
                         cash['筹资活动产生的现金流量净额(元)']
    cash_newcolumns = cash.columns.values.tolist()    
    
    #生成统计特征    
    cash_max = cash.groupby('企业编号')[cash_newcolumns[1:]].max()
    cash_min = cash.groupby('企业编号')[cash_newcolumns[1:]].min()
    cash_std = cash.groupby('企业编号')[cash_newcolumns[1:]].std()
    cash_sum = cash.groupby('企业编号')[cash_newcolumns[1:]].sum()
    cash_count = cash.groupby('企业编号')[cash_newcolumns[1]].count()
    cash_max = cash_max.reset_index()
    cash_min = cash_min.reset_index()
    cash_std = cash_std.reset_index()
    cash_sum = cash_sum.reset_index()
    cash_count = cash_count.reset_index()
    cash_max.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run_max', \
    '投资:投资活动产生的现金流量净额(元)':'invest_max', \
    '筹资活动产生的现金流量净额(元)':'finance_max', 'total_cash':'total_cash_max'}, inplace=True)
    cash_min.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run_min', \
    '投资:投资活动产生的现金流量净额(元)':'invest_min', \
    '筹资活动产生的现金流量净额(元)':'finance_min', 'total_cash':'total_cash_min'}, inplace=True)
    cash_std.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run_std', \
    '投资:投资活动产生的现金流量净额(元)':'invest_std', \
    '筹资活动产生的现金流量净额(元)':'finance_std', 'total_cash':'total_cash_std'}, inplace=True)
    cash_sum.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run_sum', \
    '投资:投资活动产生的现金流量净额(元)':'invest_sum', \
    '筹资活动产生的现金流量净额(元)':'finance_sum', 'total_cash':'total_cash_sum'}, inplace=True)
    cash_count.rename(columns={cash_newcolumns[1]:'cash_count'}, inplace=True)
    
    cash_mean = cash.groupby('企业编号')[cash_newcolumns[1:]].mean()
    cash_mean = cash_mean.reset_index()
    cash_mean.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run_mean', \
    '投资:投资活动产生的现金流量净额(元)':'invest_mean', \
    '筹资活动产生的现金流量净额(元)':'finance_mean', 'total_cash':'total_cash_mean'}, inplace=True)

    #cash = pd.merge(cash, cash_max, on='企业编号')
    cashddt = pd.merge(cash_max, cash_min, on='企业编号')
    cashddt = pd.merge(cashddt, cash_std, on='企业编号')
    cashddt = pd.merge(cashddt, cash_mean, on='企业编号')
    cashddt = pd.merge(cashddt, cash_sum, on='企业编号')
    cashddt = pd.merge(cashddt, cash_count, on='企业编号')
    
    
    #cash.rename(columns={'经营:经营活动产生的现金流量净额(元)':'run', \
    #'投资:投资活动产生的现金流量净额(元)':'invest', \
    #'筹资活动产生的现金流量净额(元)':'finance'}, inplace=True)
    #cash_newcolumns = cash.columns.values.tolist()
    
    #将-1替换为平均值
    #for col in cash_newcolumns[1:4]:
        #index = cash.loc[cash[col]==-1].index.tolist()
        #cash[col] = replace(cash[col], index, cash[col+'_mean'])

    #引入特征
    #cash['time'] = cash['日期']
    cashddt.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #利润表
    profited = pd.read_excel(filename_2)
    profit_columns = ['企业编号', '利润总额(元)', '归属母公司所有者净利润(元)', \
    '销售费用(元)', '所得税(元)', '管理费用(元)']
    profit = profited[profit_columns].copy(deep=True)
    
    for col in profit_columns[1:-1]:
        profit.fillna({col:'-1'}, inplace=True)
    
    for col in profit_columns[3:5]:
        profit[col] = profit[col].apply(lambda x: str(-1) if x=='--' else x)
    
    for col in profit_columns[1:]:
        profit[col] = profit[col].apply(lambda x: fun(x))
    
    #引入特征
    profit['sell_manage_ratio'] = profit['销售费用(元)'] / profit['管理费用(元)']
    profit_newcolumns = profit.columns.values.tolist()

    profit_max = profit.groupby('企业编号')[profit_newcolumns[1:]].max()
    profit_min = profit.groupby('企业编号')[profit_newcolumns[1:]].min()
    profit_std = profit.groupby('企业编号')[profit_newcolumns[1:]].std()
    profit_sum = profit.groupby('企业编号')[profit_newcolumns[1:]].sum()
    profit_count = profit.groupby('企业编号')[profit_newcolumns[1]].count()
    profit_max = profit_max.reset_index()
    profit_min = profit_min.reset_index()
    profit_std = profit_std.reset_index()
    profit_sum = profit_sum.reset_index()
    profit_count = profit_count.reset_index()
    
    profit_max.rename(columns={'利润总额(元)':'prosum_max', \
    '归属母公司所有者净利润(元)':'belong_max', '销售费用(元)':'sell_max', \
    '所得税(元)':'tax_max', '管理费用(元)':'manage_max', 'sell_manage_ratio':'ratio_max'}, inplace=True)
    profit_min.rename(columns={'利润总额(元)':'prosum_min', \
    '归属母公司所有者净利润(元)':'belong_min', '销售费用(元)':'sell_min', \
    '所得税(元)':'tax_min', '管理费用(元)':'manage_min', 'sell_manage_ratio':'ratio_min'}, inplace=True)
    profit_std.rename(columns={'利润总额(元)':'prosum_std', \
    '归属母公司所有者净利润(元)':'belong_std', '销售费用(元)':'sell_std', \
    '所得税(元)':'tax_std', '管理费用(元)':'manage_std', 'sell_manage_ratio':'ratio_std'}, inplace=True)
    profit_sum.rename(columns={'利润总额(元)':'prosum_sum', \
    '归属母公司所有者净利润(元)':'belong_sum', '销售费用(元)':'sell_sum', \
    '所得税(元)':'tax_sum', '管理费用(元)':'manage_sum', 'sell_manage_ratio':'ratio_sum'}, inplace=True)
    profit_count.rename(columns={profit_newcolumns[1]:'profit_count'}, inplace=True)
        
    profit_mean = profit.groupby('企业编号')[profit_newcolumns[1:]].mean()
    profit_mean = profit_mean.reset_index()
    profit_mean.rename(columns={'利润总额(元)':'prosum_mean', \
    '归属母公司所有者净利润(元)':'belong_mean', '销售费用(元)':'sell_mean', \
    '所得税(元)':'tax_mean', '管理费用(元)':'manage_mean', 'sell_manage_ratio':'ratio_mean'}, inplace=True)

    #profit = pd.merge(profit, profit_max, on='企业编号')
    profitddt = pd.merge(profit_max, profit_min, on='企业编号')
    profitddt = pd.merge(profitddt, profit_std, on='企业编号')
    profitddt = pd.merge(profitddt, profit_mean, on='企业编号')
    profitddt = pd.merge(profitddt, profit_sum, on='企业编号')
    profitddt = pd.merge(profitddt, profit_count, on='企业编号')

    #profit.rename(columns={'利润总额(元)':'prosum', \
    #'归属母公司所有者净利润(元)':'belong', '销售费用(元)':'sell', \
    #'所得税(元)':'tax', '管理费用(元)':'manage'}, inplace=True)
    #profit_newcolumns = profit.columns.values.tolist()

    #将-1替换为平均值
    #for col in profit_newcolumns[1:5]:
        #index = profit.loc[profit[col]==-1].index.tolist()
        #profit[col] = replace(profit[col], index, profit[col+'_mean'])

    #引入特征
    #profit['time'] = profited['日期']
    #profit['sell_manage_ratio'] = profit['sell'] / profit['manage']
    profitddt.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    #负债表
    debted = pd.read_excel(filename_3)
    debt_columns = ['企业编号', '资产:无形资产(元)', '资产:资产总计(元)', \
    '负债:负债合计(元)', '权益:股东权益合计(元)']
    debt = debted[debt_columns].copy(deep=True)

    for col in debt_columns[1:]:
        debt.fillna({col:'-1'}, inplace=True)
        debt.loc[debt[col]=='--', col] = str(-1)
    
    for col in debt_columns[1:]:
        debt[col] = debt[col].apply(lambda x: fun(x))
       
        
    debt_max = debt.groupby('企业编号')[debt_columns[1:]].max()
    debt_min = debt.groupby('企业编号')[debt_columns[1:]].min()
    debt_std = debt.groupby('企业编号')[debt_columns[1:]].std()
    debt_sum = debt.groupby('企业编号')[debt_columns[1:]].sum()
    debt_count = debt.groupby('企业编号')[debt_columns[1]].count()
    debt_max = debt_max.reset_index()
    debt_min = debt_min.reset_index()
    debt_std = debt_std.reset_index()
    debt_sum = debt_sum.reset_index()
    debt_count = debt_count.reset_index()
    
    debt_max.rename(columns={'资产:无形资产(元)':'inviset_max', \
    '资产:资产总计(元)':'assetsum_max', '负债:负债合计(元)':'debtsum_max', \
    '权益:股东权益合计(元)':'equsum_max'}, inplace=True)
    debt_min.rename(columns={'资产:无形资产(元)':'inviset_min', \
    '资产:资产总计(元)':'assetsum_min', '负债:负债合计(元)':'debtsum_min', \
    '权益:股东权益合计(元)':'equsum_min'}, inplace=True)
    debt_std.rename(columns={'资产:无形资产(元)':'inviset_std', \
    '资产:资产总计(元)':'assetsum_std', '负债:负债合计(元)':'debtsum_std', \
    '权益:股东权益合计(元)':'equsum_std'}, inplace=True)
    debt_sum.rename(columns={'资产:无形资产(元)':'inviset_sum', \
    '资产:资产总计(元)':'assetsum_sum', '负债:负债合计(元)':'debtsum_sum', \
    '权益:股东权益合计(元)':'equsum_sum'}, inplace=True)
    debt_count.rename(columns={debt_columns[1]:'debt_count'}, inplace=True)
    
    debt_mean = debt.groupby('企业编号')[debt_columns[1:]].mean()
    debt_mean = debt_mean.reset_index()
    debt_mean.rename(columns={'资产:无形资产(元)':'inviset_mean', \
    '资产:资产总计(元)':'assetsum_mean', '负债:负债合计(元)':'debtsum_mean', \
    '权益:股东权益合计(元)':'equsum_mean'}, inplace=True)

    #debt = pd.merge(debt, debt_max, on='企业编号')
    debtddt = pd.merge(debt_max, debt_min, on='企业编号')
    debtddt = pd.merge(debtddt, debt_std, on='企业编号')
    debtddt = pd.merge(debtddt, debt_mean, on='企业编号')
    debtddt = pd.merge(debtddt, debt_sum, on='企业编号')
    debtddt = pd.merge(debtddt, debt_count, on='企业编号')

    #debt.rename(columns={'资产:无形资产(元)':'inviset', \
    #'资产:资产总计(元)':'assetsum', '负债:负债合计(元)':'debtsum', \
    #'权益:股东权益合计(元)':'equsum'}, inplace=True)
    #debt_newcolumns = debt.columns.values.tolist()

    #将-1替换为平均值
    #for col in debt_newcolumns[1:5]:
        #index = debt.loc[debt[col]==-1].index.tolist()
        #debt[col] = replace(debt[col], index, debt[col+'_mean'])

    #引入特征
    #debt['time'] = debted['日期']
    debtddt.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################
 
    #工商基本信息表
    informated = pd.read_excel(filename_4)
    informat_columns = ['企业编号', '注册资本（万元）']
    informat = informated[informat_columns].copy(deep=True)                           
            
    informat.rename(columns={'企业编号':'id'}, inplace=True)
    ###########################################################################

    finance_feature = pd.merge(cashddt, profitddt, on='id')
    finance_feature = pd.merge(finance_feature, debtddt, on='id')
    finance_feature = pd.merge(finance_feature, informat, on='id')

    return finance_feature



#filename_1现金流量表文件路劲 
#filename_2利润表文件路劲
#filename_3负债表文件路劲
if __name__ == "__main__":
    filename_1='../现金流量表.xlsx'
    filename_2='../利润表.xlsx'
    filename_3='../负债表.xlsx'
    filename_4='../工商基本信息表.xlsx'
    finance_ddt = get_finance(filename_1, filename_2, filename_3, filename_4)
    finance_ddt.to_csv('../feature_finance.csv', index=False)








