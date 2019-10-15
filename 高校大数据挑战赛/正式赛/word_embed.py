from __future__ import print_function

from gensim.models import Word2Vec
from keras.preprocessing import text, sequence
from glove import Glove
from glove import Corpus
import pandas as pd
import numpy as np
import os

def glove_embed(texts,victor_size):
    corpus_model = Corpus()
    corpus_model.fit(texts, window=5, ignore_mising=False)

    glove = Glove(no_components= victor_size, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=20,
              no_threads=1, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    glove.save('embed_model/glove.embed_model')
    return glove

def word2vec_embed(texts,victor_size):
    word2vec = Word2Vec(texts,size= victor_size, window=5, iter=10, workers=11, seed=2019, min_count=0)
    word2vec.save('embed_model/word2vec.embed_model')
    return word2vec

def w2v_pad(df_train, df_test, maxlen_, victor_size):
    tokenizer = text.Tokenizer(num_words=500000, lower=False, filters="")

    data = pd.concat((df_train, df_test),ignore_index=True)
    tokenizer.fit_on_texts(data["text"].values.to_list())

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train["text"].values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test["text"].values), maxlen=maxlen_)

    word_index = tokenizer.word_index


    nb_words = len(word_index)
    print(nb_words)
    texts = [docs.split(' ') for docs in data["text"].values]
    word2vec_path = 'embed_model/word2vec.embed_model'
    glove_path = 'embed_model/glove.embed_model'
    if not os.path.exists(word2vec_path):
        word2vec_model = word2vec_embed(texts, victor_size)
    else:
        word2vec_model = Word2Vec.load(word2vec_path)
    if not os.path.exists(glove_path):
        glove_model = glove_embed(texts, victor_size)
    else:
        glove_model = Glove.load(glove_path)


    word2vec_count = 0
    embedding_word2vec_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_vector = word2vec_model[word] if word in word2vec_model else None
        if embedding_vector is not None:
            word2vec_count += 1
            embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_word2vec_matrix[i] = unk_vec

    glove_count = 0
    embedding_glove_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector = glove_model.word_vectors[glove_model.dictionary[word]] if word in glove_model else None
        if embedding_glove_vector is not None:
            glove_count += 1
            embedding_glove_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_glove_matrix[i] = unk_vec

    embedding_matrix = np.concatenate((embedding_word2vec_matrix, embedding_glove_matrix), axis=1)

    print(embedding_matrix.shape, train_.shape, test_.shape, word2vec_count * 1.0 / embedding_matrix.shape[0],
          glove_count * 1.0 / embedding_matrix.shape[0])
    return train_, test_, word_index, embedding_matrix

