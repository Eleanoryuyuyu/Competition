from 高校大数据挑战赛.预赛.text_process import *

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    train_data = pd.read_csv('data/train.csv', lineterminator='\n')
    test_data = pd.read_csv('data/test.csv', lineterminator='\n')
    labels = train_data['label'].values
    id = np.array(test_data['ID'].values.tolist())

    train = text_process(train_data['review'])
    test = text_process(test_data['review'])

    train_vec,test_vec = tfidf(train,test,1)
    print(train_vec.shape,test_vec.shape)

    split_index = int(0.8*len(train_vec))
    split_data = train_vec[:split_index]
    test_data = train_vec[split_index:]
    split_label = labels[:split_index]
    test_label = labels[split_index:]

    MNB = MultinomialNB(alpha=0.5)
    SG = SGDClassifier(loss="log",tol=1e-4,random_state=2019)
    LR = LogisticRegression(tol=1e-4,random_state=2019)
    clfs = [LR, SG, MNB]

    n_fold = 5
    sfd = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2019)
    totalpreds = np.zeros(test_data.shape[0])
    dataset_blend_train = np.zeros((split_data.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_data.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        print('training model:', j + 1)
        dataset_blend_test_j = np.zeros((test_data.shape[0], n_fold))
        for i, (train_index, val_index) in enumerate(sfd.split(split_data, split_label)):
            X_train = split_data[train_index]
            Y_train = split_label[train_index]
            x_test = split_data[val_index]
            y_test = split_label[val_index]
            print(i + 1, 'fold training')

            clf.fit(X_train, Y_train)
            y_submission = clf.predict_proba(x_test)[:, 1]
            dataset_blend_test_j[:, i] = clf.predict_proba(test_data)[:, 1]

            dataset_blend_train[val_index, j] = y_submission

        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    #predicts = [y for x in predicts for y in x]
    SG.fit(dataset_blend_train, split_label)
    totalpreds = SG.predict_proba(dataset_blend_test)[:, 1]
    auc = roc_auc_score(test_label, totalpreds)
    print(auc)


    #result = pd.DataFrame({'ID':id, 'Pred':totalpreds})
    #result.to_csv('result.csv',index=False)

main()



