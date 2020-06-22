from typing import List
import torch

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return idx.item()
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))

class Viterbi:
    def __init__(self,s_to_idx, v_to_idx, tran_matrix, emit_matrix):
        self.s_to_idx = s_to_idx
        self.v_to_idx = v_to_idx
        self.tran_matrix = torch.Tensor(tran_matrix).transpose(0,1)
        self.emit_matrix = torch.Tensor(emit_matrix).transpose(0,1)
        self.state_size = len(s_to_idx)

    def viterbi(self, init_state, v_seq):
        backpointers = []
        # 在对数空间初始化维特比变量
        res = []
        init_state=torch.Tensor(init_state)
        for i, s in enumerate(init_state):
            v = self.v_to_idx[v_seq[0]]
            tmp = torch.log(s)+torch.log(self.emit_matrix[v][i])
            res.append(tmp)

        del init_state
        init_vvars = torch.stack(res)

        forward_var = init_vvars
        for v in v_seq[1:]:
            bptrs_t =[]
            viterbivars_t = []
            v_index = self.v_to_idx[v]
            for s in range(self.state_size):
                next_tag_var = forward_var+torch.log(self.tran_matrix[s])
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[best_tag_id])
            forward_var = (torch.Tensor(viterbivars_t)+torch.log(self.emit_matrix[v_index]))
            backpointers.append(bptrs_t)
        # 终结
        terminal_var = forward_var
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[best_tag_id]

        # 回溯解码
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_path.reverse()
        return torch.exp(path_score), best_path



def toIdx(l:List):
    return {e:i for i, e in enumerate(l)}
def main():

    states = [ "健康", "发烧"]
    observations = ["正常","冷", "头晕"]
    tran_matrix = torch.Tensor([[0.7, 0.3], [0.4, 0.6]]) #A_ij
    emit_matrix = torch.Tensor([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    init_state = [0.6, 0.4]
    observation_seq = ["正常","冷", "头晕"]
    viterbi = Viterbi(toIdx(states), toIdx(observations), tran_matrix, emit_matrix)
    maxpro, path = viterbi.viterbi(init_state, observation_seq)
    print("最大概率为：{}".format(maxpro))
    print("最大概率下路径为：")
    pt = ""
    for i in path:
        pt += states[i] + "->"
    print(pt)
main()