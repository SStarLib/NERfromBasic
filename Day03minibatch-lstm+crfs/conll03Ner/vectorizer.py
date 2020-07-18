import numpy as np
from vocab import Vocabulary, TokenVocabulary
from itertools import chain
from collections import Counter


def creatVocab(datalist, is_tags):
    vocab = Vocabulary() if is_tags else TokenVocabulary()
    word_counts = Counter(chain(*datalist))
    valid_words = [w for w, d in word_counts.items()]
    valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
    valid_words += ['<pad>']
    for token in valid_words:
        vocab.add_token(token)
    if not is_tags:
        unk_index = vocab.add_token('<unk>')
        vocab.set_unk_index(unk_index)
    return vocab
class ConllVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, token_vocab, tag_vocab, max_seq_len=140):

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.pad = '<pad>'
        self.unk = '<unk>'

        self.max_seq_len = max_seq_len

    def _vectorize(self, seq, flag="token", vector_length=-1):
        """ 核心代码，将输入的token，和tag向量化"""
        vocab = self.token_vocab if flag == "token" else self.tag_vocab
        indices=[vocab.lookup_token(token) for token in seq]
        vec_len = len(indices)

        if vector_length < -1:
            vector_length = len(indices)
        pad_index = vocab.lookup_token(self.pad)
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = pad_index

        return vector, vec_len
    def vectorize(self, sample):
        token_vector, token_len = self._vectorize(sample[0], "token", self.max_seq_len)
        tag_vector, tag_len = self._vectorize(sample[1], "tag", self.max_seq_len)

        return {"token_vector": token_vector,
                "tag_vector": tag_vector,
                "seq_len": token_len}
    @classmethod
    def from_dataset(cls, dataset):
        tokenset = []
        tagset=[]
        for token, tag in dataset:
            tokenset.append(token)
            tagset.append(tag)
        token_vocab = creatVocab(tokenset, False)
        tag_vocab = creatVocab(tagset, True)
        return cls(token_vocab, tag_vocab)

    @classmethod
    def from_serializable(cls, contents):
        token_vocab = TokenVocabulary.from_serializable(contents["token_vocab"])
        tag_vocab = Vocabulary.from_serializable(contents["tag_vocab"])

        return cls(token_vocab=token_vocab,
                   tag_vocab=tag_vocab,
                   max_seq_len=contents["max_seq_len"])

    def to_serializable(self):
        return {"token_vocab": self.token_vocab.to_serializable(),
                "tag_vocab": self.tag_vocab.to_serializable(),
                "max_seq_len": self.max_seq_len}
