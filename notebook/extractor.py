import re
import MeCab
import numpy as np
import copy
from stop_words import create_stopwords

class Extractor:
    stopwords = create_stopwords()
    def __init__(self, documents, replaced=True, sw=True, lower=True):
        self.mecab = MeCab.Tagger('-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -u /usr/share/mecab/dic/ipadic/user.dic')
        self.n_samples = len(documents)
        parts_of_speech = ['その他', 'フィラー', '副詞', '助動詞', '助詞', '動詞', '名詞', '形容詞', '感動詞', '接続詞', '接頭詞', '記号', '連体詞']
        if replaced:
            documents = [[re.sub(r'[!-/:-@[-`{-~]', ' ', re.sub(u'[■-♯・、。°δ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳【】《》σ×θΘ“↓→↑←『「」』≦≧≠≒＊※μµφΦ]', ' ', tx)) for tx in line] for line in documents]
        if lower:
            documents = [[tx.lower() for tx in line] for line in documents]
        self.ps_rate_ = np.zeros((self.n_samples, len(parts_of_speech)))
        self.extracted = {}
        for i, ps in enumerate(parts_of_speech):
            ps_document = [[] for _ in range(self.n_samples)]
            for j, doc in enumerate(documents):
                for line in doc:
                    node = self.mecab.parseToNode(line)
                    while node:
                        w = node.feature.split(',')[-3]
                        if ps!=node.feature.split(',')[0] or w in ['NaN', '*']:
                            node = node.next
                            continue
                        if (not sw) or (not w in Extractor.stopwords):
                            ps_document[j].append(node.feature.split(',')[-3])
                            self.ps_rate_[j][i] += 1
                        node = node.next
            self.extracted[ps] = ps_document
        for i in range(self.n_samples):
            self.ps_rate_[i] /= self.ps_rate_[i].sum()
    def extract(self, selection, number_trim=True):
        documents_selected = []
        for i in range(self.n_samples):
            ps_document = []
            for ps in selection:
                ps_document += self.extracted[ps][i]
            documents_selected.append(ps_document)
        if number_trim:
            documents_selected = [[re.sub('[0-9]+', '', w) for w in doc if not w.isdigit()] for doc in documents_selected]
        self.documents_raw = copy.deepcopy(documents_selected)
        n_col_documents = np.max([len(doc) for doc in documents_selected])
        for i in range(self.n_samples):
            documents_selected[i] += [''] * (n_col_documents - len(documents_selected[i]))
        return documents_selected
