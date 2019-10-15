
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

def text_process(raw_text):
    text_list = []
    for text in raw_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text)

        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)

        # 恢复特殊符号的英文单词
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')
        # text = text.split()

        text_list.append(text)
    return text_list
def tfidf(train,test,ngram):
    all_comment_list = list(train+test)
    text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', token_pattern=r'\w{1,}',
                                  max_features=9500, ngram_range=(1, ngram), analyzer='word',stop_words='english')
    vec = text_vector.fit_transform(all_comment_list)
    train_vec = vec[:len(train)]
    test_vec = vec[len(train):]
    s_train = train_vec.sum(axis=1)[:,0]
    m_train = train_vec.mean(axis=1)[:,0]
    s_test = test_vec.sum(axis=1)[:, 0]
    m_test = test_vec.mean(axis=1)[:, 0]
    train_vec = sparse.hstack((train_vec, s_train))
    train_vec = sparse.hstack((train_vec, m_train))
    test_vec = sparse.hstack((test_vec, s_test))
    test_vec = sparse.hstack((test_vec, m_test))

    return train_vec.toarray(),test_vec.toarray()

def tfidf_naive(train,test):
    all_comment_list = list(train+test)

    text_vector = TfidfVectorizer(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english')

    vec = text_vector.fit_transform(all_comment_list)
    # print(len(train))
    train_vec = vec[:len(train)]
    test_vec = vec[len(train):]
    return train_vec,test_vec

