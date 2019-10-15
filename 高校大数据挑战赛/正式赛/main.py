import pandas as pd
from keras.callbacks import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
from 高校大数据挑战赛.正式赛.model.nn_model import *
from 高校大数据挑战赛.正式赛.word_embed import *

def read_data():
    path = 'data/train_data.csv'
    reader = pd.read_csv(path, sep=',', iterator=True, header=None,
                         names=['query_id', 'query','title_id', 'title', 'label'])
    chunkSize = 10000
    loop = True
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df = pd.concat(chunks, ignore_index=True)
    df["text"] = df["query"] + df["title"]
    print(df.head())
    return df

df = read_data()
split_index = df.shape[0] * 0.8
train = df[:split_index]
test = df[split_index:]
word_seq_len = 300
victor_size = 200
train_, test_, word_index, word_embedding = w2v_pad(train, test, word_seq_len, victor_size)

def model_train(name):
    train_label = train["label"].values
    test_label = test["label"].values

    # 模型
    kf = KFold(n_splits=10, shuffle=True, random_state=2019).split(train_)
    train_model_pred = np.zeros((train_.shape[0], 2))
    test_model_pred = np.zeros((test_.shape[0], 2))

    select_model = eval(name)
    model = select_model(word_seq_len,word_embedding)
    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = train_[train_fold, :], train_[test_fold, :]
        y_train, y_valid = train_label[train_fold], train_label[test_fold]

        print(i, 'fold')

        the_path =  name + "_"

        early_stopping = EarlyStopping(monitor='val_acc', patience=6)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(the_path + str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True,
                                     mode='max', save_weights_only=True)
        if not os.path.exists(the_path + str(i) + '.hdf5'):
            print("error")
            model.fit(X_train, y_train,
                      epochs=100,
                      batch_size=64,
                      validation_data=(X_valid, y_valid),
                      callbacks=[early_stopping, plateau, checkpoint],
                      verbose=2)

        model.load_weights(the_path + str(i) + '.hdf5')

        print(name + ": valid's aucscore: %s" % roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1]))

        train_model_pred[test_fold, :] = model.predict_proba(X_valid)
        test_model_pred += model.predict_proba(test_)

        del model
        gc.collect()
        K.clear_session()
    # 线下测试
    test_model_pred = test_model_pred/10
    print(name + ": offline test aucscore: %s" % roc_auc_score(test_label, test_model_pred[:,1]))


