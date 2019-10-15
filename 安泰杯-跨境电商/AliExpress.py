import pandas as pd
from operator import itemgetter
from math import *
from collections import defaultdict


def read_data():
    train_df = pd.read_csv("data/Antai_AE_round2_train_20190813.csv")
    test_df = pd.read_csv("data/Antai_AE_round2_test_20190813.csv")
    print(train_df.shape, test_df.shape)
    return train_df, test_df


def get_only_buy(test_df):
    buy = test_df[test_df["buy_flag"] == 1]["buyer_admin_id"].unique().tolist()
    click = test_df[test_df["buy_flag"] == 0]["buyer_admin_id"].unique().tolist()
    buy_index = set(buy).intersection(set(click))
    only_click = test_df[test_df["buy_flag"] == 0].set_index("buyer_admin_id").drop(buy_index)
    only_click = only_click.reset_index()
    test = pd.concat([test_df[test_df["buy_flag"] == 1], only_click], ignore_index=True)
    return test


def drop_rep(data):
    data = data.sort_values(by=["buyer_admin_id", "irank", "buy_flag"], ascending=[True, True, False])
    data = data.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    return data


def get_drop_items(train_df, test_df):
    train_items = train_df["item_id"].unique()
    test_items = test_df["item_id"].unique()
    drop_items = list(set(test_items).difference(set(train_items)))
    print(len(train_items), len(test_items), len(drop_items))
    return drop_items


def df2_item_dict(df):
    df = df.sort_values(by=["buyer_admin_id", "irank"])
    df_group = df.groupby("buyer_admin_id").agg({"item_id": lambda x: list(x.unique())})
    df_group.columns = ["item_list"]
    data_dict = df_group.to_dict(orient='index')
    data_dict = {k: v["item_list"] for k, v in data_dict.items()}
    return data_dict


def df2_user_dict(df):
    df = df.sort_values(by=["buyer_admin_id", "irank"])
    df_group = df.groupby("item_id").agg({"buyer_admin_id": lambda x: list(x.unique())})
    df_group.columns = ["user_list"]
    print(df_group.head())
    data_dict = df_group.to_dict(orient='index')
    data_dict = {k: v["user_list"] for k, v in data_dict.items()}
    return data_dict


def UserCF(train):
    print("start calculate item_sim")
    N = defaultdict(lambda: 0)
    C = defaultdict(lambda: dict())
    for item, users in train.items():
        for u in users:
            N[u] += 1
            for v in users:
                if v == u:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1 / log(1 + len(users) * 1.0)
    W = defaultdict(lambda: dict())
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / sqrt(N[u] * N[v])
    return W


def GetRecommendation(user, items_t, W):
    items = items_t.copy()
    recom = [i for i in items if i not in drop_items]

    if user not in W or len(recom) >= 30:
        recom = recom[:30]
    else:
        sim_items = defaultdict(lambda: 0)
        sim_users = sorted(W[user].items(), key=itemgetter(1), reverse=True)[:100]
        for v, wuv in sim_users:
            for i in unmature_user_dict[v]:
                if i not in items and i not in drop_items:
                    sim_items[i] += wuv
        sim_items = sorted(sim_items.items(), key=itemgetter(1), reverse=True)[:80]
        sim_items = [item[0] for item in sim_items]

        m = 30 - len(recom)
        r = m if m <= len(sim_items) else len(sim_items)
        for j in range(r):
            it = sim_items.pop(0)
            recom.append(it)

    return recom


def get_item_list(test_dict):
    recom_dic = {}
    for user, items in test_dict.items():
        recom_items = GetRecommendation(user, items, sim_user)
        recom_dic[user] = recom_items
    return recom_dic


def get_hots(unmature):
    temp = unmature.copy()
    item_cnts = temp.groupby(['item_id']).size().reset_index()
    item_cnts.columns = ['item_id', 'cnts']
    item_cnts = item_cnts.sort_values('cnts', ascending=False)
    hot_items = item_cnts['item_id'].values.tolist()
    return hot_items


def add_hots(recom_dict, hot_items):
    recom_re = recom_dict.copy()
    for u, items in recom_re.items():
        if len(items) == 30:
            items = items
        elif len(items) < 30:
            print(u)
            m = 30 - len(items)
            for i in range(m):
                for j in range(50):
                    it = hot_items.pop(0)
                    if it not in items:
                        items.append(it)
                        break
        elif len(items) > 30:
            items = items[:30]
        recom_re[u] = items
    return recom_re


train_df, test_df = read_data()
unmature_df = train_df[(train_df["country_id"] == "yy") | (train_df["country_id"] == "zz")]
test = get_only_buy(test_df)
test = drop_rep(test)
test_tmp = drop_rep(test_df)
unmature = drop_rep(unmature_df)
train_data = pd.concat([unmature, test_tmp], ignore_index=True)
drop_items = get_drop_items(train_df, test_df)
test_dict = df2_item_dict(test)
unmature_user_dict = df2_item_dict(train_data)
print("test_dict size :", len(test_dict))
train_dict = df2_user_dict(train_data)
print("train_dict size:", len(train_dict))
sim_user = UserCF(train_dict)
recom_dict = get_item_list(test_dict)
print(len(recom_dict))
hot_items = get_hots(unmature)
recom_re = add_hots(recom_dict, hot_items)
recom_df = pd.DataFrame(recom_re).T
recom_df = recom_df.reset_index()
recom_df.to_csv('submit.csv', header=None, index=None)
