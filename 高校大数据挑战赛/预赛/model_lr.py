from 高校大数据挑战赛.预赛.text_process import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



def main():
    train_data = pd.read_csv('data/train.csv',lineterminator='\n')
    test_data = pd.read_csv('data/test.csv',lineterminator='\n')

    labels = train_data['label'].values

    train = text_process(train_data['review'].values)
    test = text_process(test_data['review'].values)
    train_vec,test_vec = tfidf(train,test,1)
    print(train_vec.shape,test_vec.shape)

    stf = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
    model_LR = LogisticRegression()

    aucs = []
    for train_index,test_index in stf.split(train_vec,labels):
        x_train = train_vec[train_index]
        y_train = labels[train_index]
        x_test = train_vec[test_index]
        y_test = labels[test_index]

        model_LR.fit(x_train, y_train)
        y_pred = model_LR.predict_proba(x_test)
        auc = roc_auc_score(y_test,y_pred[:,1])
        print(auc)
        aucs.append(auc)

    print(sum(aucs)/10.0)



main()

