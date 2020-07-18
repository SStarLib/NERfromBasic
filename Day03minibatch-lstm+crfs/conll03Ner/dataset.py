from torch.utils.data import Dataset, DataLoader
import json
from vectorizer import ConllVectorizer
import torch

class ConllDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self._vectorizer = vectorizer

        self.train_ = self.data['train']
        self.train_size = len(self.train_)

        self.val_ = self.data['valid']
        self.validation_size = len(self.val_)

        self.test_ = self.data['test']
        self.test_size = len(self.test_)

        self._lookup_dict = {'train': (self.train_, self.train_size),
                             'val': (self.val_, self.validation_size),
                             'test': (self.test_, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, data):
        """Load dataset and make a new vectorizer from scratch

        Args:
            surname_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        train_subset = data['train']
        return cls(data, ConllVectorizer.from_dataset(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, data, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(data, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return ConllVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        sample = self._target_[index]


        vector_dict = self._vectorizer.vectorize(sample)

        return {"token_vec": vector_dict["token_vector"],
                "tag_vec": vector_dict["tag_vector"],
                "seq_len": vector_dict["seq_len"]}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_conll_batches(dataset, batch_size, shuffle=True,
                           drop_last=True, device="cpu"):
    """
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param drop_last:
    :param device:
    :return:
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['seq_len'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        max_len=torch.max(data_dict['seq_len']).item()
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
            if name != 'seq_len':
                out_data_dict[name] = out_data_dict[name][:, :max_len]
        yield out_data_dict
