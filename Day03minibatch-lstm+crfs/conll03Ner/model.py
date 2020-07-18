import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

begin_seq_token="<BEGIN>"
end_seq_token="<END>"
################################### 工具函数
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
################################### 工具函数 end
# 模型
class BiLSTM_CRF(nn.Module):

    def __init__(self, token_vocab, tag_vocab, batch_size,
                 dropout=0.5, embedding_dim=256,
                 hidden_dim=256, pretrained_embedding=None,
                 padding_idx=0, num_layers=1,):
        super(BiLSTM_CRF, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.token_vocab=token_vocab
        self.tag_vocab = tag_vocab
        self.pad = self.token_vocab.pad_token

        self.tagset_size = len(tag_vocab)
        self.begin_tag_idx=tag_vocab.lookup_token('<start>')
        self.end_tag_idx = tag_vocab.lookup_token('<end>')

        if pretrained_embedding is None:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim)
        else:
            self.word_embeds = nn.Embedding(len(self.token_vocab), embedding_dim,
                                            _weight=pretrained_embedding)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transition = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transition.data[self.begin_tag_idx, :] = -10000
        self.transition.data[:, self.end_tag_idx] = -10000

        self.hidden = self.init_hidden(num_layers, batch_size)

    def init_hidden(self,num_layers, batch_size):
        return (torch.randn(2*num_layers, batch_size, self.hidden_dim // 2, device=self.device),
                torch.randn(2*num_layers, batch_size, self.hidden_dim // 2, device=self.device))

    def _forward_alg(self, feats, mask):
        """前向算法
        :param feats: [b_s, seq_len, tag_size]
        :param mask: [b_s, seq_len]
        :return:
        """
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((feats.size(0), self.tagset_size), -10000., device=self.device)    #[b_s, tag_size]
        # START_TAG has all of the score.along dim=1,
        init_alphas[:, self.begin_tag_idx]=0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var_list=[]
        forward_var_list.append(init_alphas)
        d = torch.unsqueeze(feats[:,0], dim=1)  #[b_s, 1, tag_size]
        for feat_index in range(1, feats.size(1)):
            n_unfinish = mask[:, feat_index].sum()
            d_uf = d[:n_unfinish] #[uf, 1, tag_size]
            emit_and_transition = feats[: n_unfinish, feat_index].unsqueeze(dim=1)+self.transition #[uf,tag_size,tag_size]
            log_sum = d_uf.transpose(1, 2)+emit_and_transition  #[uf, tag_size, tag_size]
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  #[uf, 1, tag_size]
            log_sum = log_sum - max_v   #[uf, tag_size, tag_size]
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1) # [uf, 1, tag_size]
            d = torch.cat((d_uf, d[n_unfinish:]), dim=0)
        d = d.squeeze(dim=1)    #[b_s, tag_size]
        max_d = d.max(dim=-1)[0]  # [b_s]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # [b_s]
        return d

    def _get_lstm_features(self, embedded_vec, seq_len):
        """
        用lstm学习emit_score, h_d: hidden_dim
        max_seq_len:全部数据集最大句子长度，seq_len:batch内最大句子长度
        :param embedded_vec: [max_seq_len, b_s, e_d]
        :param seq_len: [b_s]
        :return:
        """
        # 初始化 h0 和 c0,可以缺省 shape:
        # ([num_layers * num_directions, batch, hidden_size],[num_layers * num_directions, batch, hidden_size])
        # self.hidden = self.init_hidden(1, seq_len.size(0))
        pack_seq = pack_padded_sequence(embedded_vec, seq_len)
        # 不初始化状态，默认初始状态都为0
        # lstm_out, self.hidden = self.lstm(pack_seq, self.hidden)
        lstm_out, self.hidden = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True) #[b_s, seq_len, h_d]
        lstm_feats = self.hidden2tag(lstm_out)  #[b_s, seq_len, tag_size]
        lstm_feats = self.dropout(lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):
        """
        计算 正常tag下的得分。需要mask掉不想算的部分
        :param feats:[b_s, seq_len, tag_size]
        :param tags:[b_s, seq_len]
        :param mask:[b_s, seq_len]
        :return:
        """
        score = torch.gather(feats, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)
        return total_score

    def _viterbi_decode(self, feats, mask, seq_len):
        """

        :param sentences:
        :param sen_lengths:
        :return:
        """
        batch_size = feats.size(0)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(feats[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, seq_len[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + feats[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # [b_s, tag_size
        score, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return score, tags

    def neg_log_likelihood(self, token_vec, tag_vec, seq_len):
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)  # [b_s, max_seq_len]
        token_vec = token_vec.transpose(0, 1)  # [max_seq_len, b_s]
        embedded_vec = self.word_embeds(token_vec)  # [max_seq_len, b_s, e_d]
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(embedded_vec, seq_len)  # [b_s, seq_len, tag_size]

        forward_score = self._forward_alg(feats, mask)  # [b_s]
        gold_score = self._score_sentence(feats, tag_vec, mask)  #[b_s]
        return forward_score - gold_score #[b_s]

    def forward(self, token_vec, tag_vec, seq_len):  # dont confuse this with _forward_alg above.
        """
        维度：seq_len:sequence length(句子长度）, b_s:batch_size（批大小）
            e_d: embedding_dim（嵌入层维度）
        :param token_vec:句子向量 [b_s, max_seq_len]
        :param tag_vec:标签向量
        :param seq_len:句子长度
        :return:
        """
        mask = (token_vec != self.token_vocab.lookup_token(self.pad)).to(self.device)  # [b_s, max_seq_len]
        token_vec = token_vec.transpose(0, 1)  # [max_seq_len, b_s]
        embedded_vec = self.word_embeds(token_vec) #[max_seq_len, b_s, e_d]
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(embedded_vec, seq_len)     #[b_s, seq_len, tag_size]

        # tag_vec = tag_vec[:, :lstm_feats.size(1)]
        mask = mask[:, :lstm_feats.size(1)]
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, mask, seq_len)
        return score, tag_seq

    @property
    def device(self):
        return self.word_embeds.weight.device
