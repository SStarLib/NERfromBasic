from tqdm import tqdm
import os


class Conll03Reader:
    def read(self, data_path):
        data_parts = ['train', 'valid', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = os.path.join(data_path, data_part + extension)
            dataset[data_part] = self.read_file(str(file_path))
        return dataset

    def read_file(self, file_path):
        samples = []
        tokens = ['<start>']
        tag = ['<start>']
        with open(file_path, 'r', encoding='utf-8') as fb:
            for line in fb:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    # 去除数据头
                    pass
                elif line == '':
                    # 一句话结束
                    if len(tokens) > 1:
                        samples.append((tokens + ['<end>'], tag + ['<end>']))
                        tokens = ['<start>']
                        tag = ['<start>']
                else:
                    # 数据分割，只要开头的词和最后一个实体标注。
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    tag.append(contents[-1])
        return samples

def predata(input_path="./conll2003_v2"):
    ds_rd = Conll03Reader()
    condata = ds_rd.read(input_path)
    return condata
