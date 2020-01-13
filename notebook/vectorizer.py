import re
import time
import numpy as np
from sklearn import mixture
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from tqdm import tqdm
import gensim

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('model/model_fasttext_neo.vec', binary = False)

class Vectorizer:
    word_vecs = word_vectors
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.word_vec_dim = Vectorizer.word_vecs.vector_size
        self.dr_ = None
        self.n_samples = None
        self.X = None

    def fit(self, X, y=None):
        return self

    def add_feature(self, X):
        self.add_letter_len(X)
        self.add_used_rate(X)
        self.add_char_rate(X)

    def add_letter_len(self, X):
        word_len = []
        for i, doc in enumerate(X):
            cnt = 0
            for w in doc:
                cnt += len(w)
            if len(doc)==0:
                word_len.append(0.0)
            else:
                word_len.append(cnt/len(doc))
        word_len = np.array(word_len).reshape(-1, 1)
        self.dr_ = np.hstack([self.dr_, word_len])

    def add_used_rate(self, X):
        used_rate = []
        corpus = set(Vectorizer.word_vecs.index2word)
        for i, doc in enumerate(X):
            cnt = 0
            for w in doc:
                if w in corpus:
                    cnt += 1
            if len(doc)==0:
                used_rate.append(0)
            else:
                used_rate.append(cnt/len(doc))
        used_rate = np.array(used_rate).reshape(-1, 1)
        self.dr_ = np.hstack([self.dr_, used_rate])

    def add_char_rate(self, X):
        char_rate = []
        hira = re.compile('[ぁ-ゟ]+')
        digit = re.compile('[0-9]+')
        kanji = re.compile('[\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+')
        for doc in X:
            num_hira = 0
            num_digit = 0
            num_kanji = 0
            for w in doc:
                num_hira += int(hira.search(w)!=None)
                num_digit += int(digit.search(w)!=None)
                num_kanji += int(kanji.fullmatch(w)!=None)
            if len(doc)==0:
                char_rate.append([0, 0, 0])
            else:
                char_rate.append([num_hira/len(doc), num_digit/len(doc), num_kanji/len(doc)])
        self.dr_ = np.hstack([self.dr_, np.array(char_rate)])

class SCDV(Vectorizer):
    def __init__(self, n_components=3, random_state=0):
        super().__init__(random_state)
        self.n_components = n_components

    def transform(self, X):
        self.X = X
        self.n_samples = len(X)

    def idf(word):
        df = np.sum(np.array([int(word in d) for d in X], dtype="float32"))
        return np.log((self.n_samples+1) / (df+1.0)) + 1

        ex_words = set(Vectorizer.word_vecs.index2word)
        used_words = set([w for doc in X for w in doc])
        self.ex_used_words = list(ex_words & used_words)

        used_word_vectors = np.array([Vectorizer.word_vecs[w] for w in self.ex_used_words])
        gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type='tied', max_iter=50, random_state=self.random_state)
        gmm.fit(used_word_vectors)
        word_probs = gmm.predict_proba(used_word_vectors)

        word_cluster_vectors = np.zeros(shape=(len(self.ex_used_words), self.n_components*self.word_vec_dim))
        for i, used_word in enumerate(self.ex_used_words):
            tmp = np.array([])
            for j in range(self.n_components):
                tmp = np.hstack([tmp, idf(used_word)*word_probs[i][j]*Vectorizer.word_vecs[used_word]])
            word_cluster_vectors[i] = tmp
        self.dr_ = np.zeros(shape=(self.n_samples, self.n_components*self.word_vec_dim))
        cnts = np.zeros(self.n_samples)
        for i, doc in enumerate(X):
            for w in doc:
                if not w in self.ex_used_words:
                    continue
                self.dr_[i] += word_cluster_vectors[self.ex_used_words.index(w)]
                cnts[i] += 1
        for i in range(self.n_components):
            if np.linalg.norm(self.dr_[i])==0:
                continue
          # doc_vectors[i] = doc_vectors[i]/np.linalg.norm(doc_vectors[i])
            self.dr_[i] /= cnts[i]
        self.add_feature(X)
        return self.dr_

class SWEM(Vectorizer):
    def __init__(self, pooling="average", random_state=0):
        super().__init__(random_state)
        self.pooling = pooling
        self.idf = {}

    def transform(self, X):
        self.X = X
        self.n_samples = len(X)
        ex_words = set(Vectorizer.word_vecs.index2word)
        used_words = set([w for doc in X for w in doc])
        self.get_idf(used_words)
        self.ex_used_words = list(ex_words & used_words)
        self.dr_ = np.zeros((len(X), self.word_vec_dim))
        for i, doc in enumerate(tqdm(X)):
            if self.pooling == "max":
                self.dr_[i, :] = self.max_pooling(doc)
            elif self.pooling == "average":
                self.dr_[i, :] = self.average_pooling(doc)
        self.add_feature(X)
        return self.dr_

    def max_pooling(self, doc):
        if doc == []:
            return np.zeros(self.word_vec_dim)
        # https://nbviewer.jupyter.org/github/nekoumei/Comparison-DocClassification/blob/master/src/Classification_News.ipynb
        doc_vector = np.zeros((len(doc), self.word_vec_dim))
        for i, word in enumerate(doc):
            try:
                wv = Vectorizer.word_vecs[word]
            except KeyError:
                wv = np.zeros(self.word_vec_dim)
            doc_vector[i, :] = wv
        doc_vector = np.max(doc_vector, axis=0)
        return doc_vector

    def average_pooling(self, doc):
        doc_vector = np.zeros(self.word_vec_dim)
        cnt = 0
        for word in doc:
            if word in Vectorizer.word_vecs.vocab.keys():
                wv = Vectorizer.word_vecs[word]
                doc_vector += self.tfidf(doc, word) * wv
                cnt += 1
        if cnt!=0:
            doc_vector /= cnt
        return doc_vector

    def tfidf(self, doc, word):
        cnt_w = 0
        for w in doc:
            if w == word:
                cnt_w += 1
        tf = 1+np.log(cnt_w)
        return tf*self.idf[word]

    def get_idf(self, corpus):
        for word in corpus:
            cnt = 0
            for doc in self.X:
                if word in doc:
                    cnt += 1
            self.idf[word] = np.log(1+self.n_samples/(1+cnt))

class TF_IDF_Transfer:
    def __init__(self):
        self.tokenizer = None
    def fit(self, X, y=None):
        self.tokenizer = Tokenizer().fit_on_texts(X)
    def transform(self, X):
        tf_idf_X = self.tokenizer.texts_to_matrix(X, "tfidf")
