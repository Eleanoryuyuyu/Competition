from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def add_feature(data):
    data["num_words"] = data["text"].apply(lambda x: len(str(x).split()))
    data["num_unique_words"] = data["text"].apply(lambda x: len(set(str(x).split())))
    data["title_len"] = data["title"].apply(lambda x: len(str(x).split()))
    data["query_len"] = data["query"].apply(lambda x: len(str(x).split()))
    return data


def tfidf(data, ngram):
    all_comment_list = data
    text_vector = TfidfVectorizer(sublinear_tf=True, token_pattern=r'\w{1,}',
                                  max_features=1200, ngram_range=(1, ngram), analyzer='word')
    vec = text_vector.fit_transform(all_comment_list)
    s_vec = vec.sum(axis=1)[:, 0]
    m_vec = vec.mean(axis=1)[:, 0]
    data_vec = sparse.hstack((vec, s_vec))
    data_vec = sparse.hstack((data_vec, m_vec))

    #     sort = np.argsort(vec.toarray(), axis=1)[:, -5:]
    #     words = text_vector.get_feature_names()
    #     key_words = pd.Index(words)[sort].values

    return data_vec.toarray()