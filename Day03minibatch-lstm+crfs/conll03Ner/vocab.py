class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self,token_to_idx=None, pad_token='<pad>'):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        self.pad_token = pad_token

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx,
                'pad_token': self.pad_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class TokenVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, pad_token='<pad>',
                 unk_token='<unk>', unk_index=0):

        super(TokenVocabulary, self).__init__(token_to_idx, pad_token)
        self._unk_token = unk_token
        self.unk_index=unk_index

    def set_unk_index(self,unk_index):
        self.unk_index=unk_index

    def to_serializable(self):
        contents = super(TokenVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'unk_index': self.unk_index})
        return contents
    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]