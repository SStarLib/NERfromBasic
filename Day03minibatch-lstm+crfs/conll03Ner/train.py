from dataset import generate_conll_batches, ConllDataset
from pre_data import predata
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import os
from argparse import Namespace
from model import BiLSTM_CRF


###########################工具函数
def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            # 'train_acc': [],
            'val_loss': [],
            # 'val_acc': [],
            'test_loss': -1,
            # 'test_acc': -1,
            'model_filename': args.model_state_file}


def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= loss_tm1:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


###########################工具函数 end

args = Namespace(data_path="./conll2003_v2",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 save_dir="model_storage/conll2003_v2/lstmcrf",
                 result_file="result.txt",
                 reload_from_files=True,
                 expand_filepaths_to_save_dir=True,
                 cuda=True,
                 seed=42,
                 learning_rate=1e-3,
                 batch_size=32,
                 num_epochs=50,
                 early_stopping_criteria=5,
                 # 超参数配置，待确定
                 glove_filepath='./glove/glove.6B.100d.txt',
                 use_glove=True,
                 dropout=0.5,
                 clip_max_norm=5.0,
                 embedding_dim=100,
                 hidden_dim=50,
                 catch_keyboard_interrupt=True)
if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)

    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)

condata = predata(args.data_path)
if args.reload_from_files and os.path.exists(args.vectorizer_file):
    # training from a checkpoint
    dataset = ConllDataset.load_dataset_and_load_vectorizer(condata,
                                                            args.vectorizer_file)
else:
    # create dataset and vectorizer

    dataset = ConllDataset.load_dataset_and_make_vectorizer(condata)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()
# Use GloVe or randomly initialized embeddings
if args.use_glove:
    words = vectorizer.token_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                       words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None

model = BiLSTM_CRF(vectorizer.token_vocab, vectorizer.tag_vocab, args.batch_size,
                   dropout=args.dropout, embedding_dim=args.embedding_dim,
                   hidden_dim=args.hidden_dim)
if args.reload_from_files and os.path.exists(args.model_state_file):
    model.load_state_dict(torch.load(args.model_state_file))
    print("Reloaded model")
else:
    print("New model")

model = model.to(args.device)
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.xavier_normal_(param.data)
    else:
        nn.init.constant_(param.data, 0)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                 factor=0.5, patience=1)
train_state = make_train_state(args)

epoch_bar = tqdm(desc='training routine',
                 total=args.num_epochs,
                 position=0)

dataset.set_split('train')
train_bar = tqdm(desc='split=train',
                 total=dataset.get_num_batches(args.batch_size),
                 position=1,
                 leave=True)
dataset.set_split('val')
val_bar = tqdm(desc='split=val',
               total=dataset.get_num_batches(args.batch_size),
               position=1,
               leave=True)

for epoch_index in range(args.num_epochs):
    train_state["epoch_index"] = epoch_index
    dataset.set_split('train')
    batch_generator = generate_conll_batches(dataset,
                                             batch_size=args.batch_size,
                                             device=args.device)
    train_loss_sum = 0
    train_batch_size = 0
    valid_loss_sum = 0
    valid_batch_size = 0

    running_loss=0
    model.train()
    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        current_batch_size =len(batch_dict["token_vec"])

        batch_loss = model.neg_log_likelihood(batch_dict["token_vec"],
                                              batch_dict["tag_vec"],
                                              batch_dict["seq_len"])  # [b_s]
        # batch_loss = model(batch_dict["token_vec"],
        #                                           batch_dict["tag_vec"],
        #                                           batch_dict["seq_len"])  # [b_s]
        loss = batch_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm)
        optimizer.step()
        train_loss_sum += batch_loss.sum().item()
        train_batch_size += current_batch_size
        running_loss=train_loss_sum/train_batch_size

        train_bar.set_postfix(loss=running_loss, epoch=epoch_index)
        train_bar.update()

    train_state['train_loss'].append(running_loss)

    dataset.set_split('val')
    batch_generator = generate_conll_batches(dataset,
                                             batch_size=args.batch_size,
                                             device=args.device)

    running_loss=0
    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        batch_loss = model.neg_log_likelihood(batch_dict["token_vec"],
                                              batch_dict["tag_vec"],
                                              batch_dict["seq_len"])  # [b_s]
        current_batch_size = len(batch_dict["token_vec"])
        valid_loss_sum+=batch_loss.sum().item()
        valid_batch_size+=current_batch_size
        running_loss=valid_loss_sum/valid_batch_size
        val_bar.set_postfix(loss=running_loss, epoch=epoch_index)
        val_bar.update()

    train_state['val_loss'].append(running_loss)
    train_state = update_train_state(args=args, model=model, train_state=train_state)
    scheduler.step(train_state['val_loss'][-1])
    if train_state['stop_early']:
        break
    train_bar.n = 0
    val_bar.n = 0
    epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'])
    epoch_bar.update()


model = BiLSTM_CRF(vectorizer.token_vocab, vectorizer.tag_vocab, args.batch_size,
                   dropout=args.dropout, embedding_dim=args.embedding_dim,
                   hidden_dim=args.hidden_dim)
model.load_state_dict(torch.load(args.model_state_file))
model = model.to(args.device)
model.eval()
dataset.set_split('test')
batch_generator = generate_conll_batches(dataset,
                                         batch_size=args.batch_size,
                                         device=args.device)
result_file = open(args.result_file, 'w')
for batch_index, batch_dict in enumerate(batch_generator):
    score, pre_tag_vec = model(batch_dict["token_vec"],
                               batch_dict["tag_vec"], batch_dict["seq_len"])
    token_vec = batch_dict["token_vec"]
    true_tag_vec = batch_dict["tag_vec"]
    for sent, true_tag, pred_tag in zip(token_vec, true_tag_vec, pre_tag_vec):
        sent, true_tag, pred_tag = sent[1: -1], true_tag[1: -1], pred_tag[1: -1]
        for token, true_tag, pred_tag in zip(sent, true_tag, pred_tag):
            result_file.write(' '.join([vectorizer.token_vocab.lookup_index(token.item()),
                                       vectorizer.tag_vocab.lookup_index(true_tag.item()),
                                       vectorizer.tag_vocab.lookup_index(pred_tag)]) + '\n')
        result_file.write('\n')
