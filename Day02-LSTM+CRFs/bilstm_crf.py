import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

class BiLSTM_CRF(nn.Module):
    """
    这是一个序列标注的问题，将句子按BIO的方式标注，Begin，实体的开始，Inside，Outside。
    把句子的开始B,I,O START_TAG, STOP_TAG,作为状态集合，对应状态序列Q
    sentence 作为观察序列 O

    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        # 把LSTM的输出映射到 tag 空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 从状态j转移到状态i的转移矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 这两句约束绝不会转移到START_TAG 和 STOP_TAG,将负分搞到足够大
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim//2).to(device),
                torch.randn(2, 1, self.hidden_dim//2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 做前向算法计算分区函数
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        # START_TAG拥有所有分数。
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0

        # Wrap in a variable so that we will get automatic backprop
        # 包装进一个变量，可以反向传播
        forward_var = init_alphas

        # Iterate through the sentence
        # 遍历句子
        for feat in feats:
            alphas_t = [] # The forward tensors at this timestep 前向张量
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # 广播发射分数：与之前的标签无关，该分数相同
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # trans_score的第i个条目是从i过渡到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # next_tag_var的第i个条目是进行log-sum-exp之前的边的值（i-> next_tag）
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                # 此标签的前向变量是所有分数的log-sum-exp。
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 给出所提供标签序列的分数
        score = torch.zeros(1).to(device)
        # 把START_TAG 拼接上
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags]).to(device)
        for i, feat in enumerate(feats):
            score = score + self.transitions[ tags[i + 1], tags[i] ] + feat[ tags[i + 1] ]
        score = score + self.transitions[ self.tag_to_ix[STOP_TAG], tags[-1] ]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        #Initialize the viterbi variables in log space
        # 在对数空间初始化维特比变量
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # step i 保存 step i-1 的 viterbi 变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step 保存该step 的backpointers
            viterbivars_t = [] # holds the viterbi variables for this step 保存该step的维特比变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # next_tag_var [i]在上一步中保存标签i的viterbi变量，以及从标签i过渡到next_tag的分数。
                # 我们不包括排放分数，因为最大值并不取决于它们（我们在下面添加它们）
                next_tag_var = forward_var+self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t)+feat).view(1, -1)
            backpointers.append(bptrs_t)
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]   # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):        # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()), ("georgia tech is a university in georgia".split(),
    "B I O O O O B".split() )]

word_to_idx = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTM_CRF(len(word_to_idx), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_idx).to(device)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long).to(device)
    print(model(precheck_sent))

# Check predictions before training
for epoch in range(300):    # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_idx).to(device)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)

        # Step3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_idx).to(device)
    print(model(precheck_sent))