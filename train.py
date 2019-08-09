import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import transformer
# from transformer.Translator import Chatbot
from dataset import Dataset
from transformer.Models import Transformer

# load config
with open("config.json", "r") as f:
    config = json.load(f)

for key, data in config.items():
    print("{}: {}".format(key, data))

# create output dir to save model, and results in
if not os.path.exists(config["output_dir"]):
    os.mkdir(config["output_dir"])

# create train dataset
train_dataset = Dataset(
    os.path.join(config["dataset_filename"], "all.bpe.src-mt"),
    os.path.join(config["dataset_filename"], "all.bpe.pe"),
    config["history_len"],
    config["response_len"])

# creat validation dataset
val_dataset = Dataset(
    os.path.join(config["dataset_filename"], "dev.bpe.src-mt"),
    os.path.join(config["dataset_filename"], "dev.bpe.pe"),
    config["history_len"],
    config["response_len"],
    train_dataset.vocab)


# set vocab:
vocab = val_dataset.vocab
train_dataset.vocab = vocab
config["vocab_size"] = len(vocab)
vocab.save_to_dict(os.path.join(config["output_dir"], "vocab.json"))

# print info
print("train_len: {}\nval_len: {}\nvocab_size: {}".format(len(train_dataset), len(val_dataset), len(vocab)))

# initialize dataloaders
data_loader_train = torch.utils.data.DataLoader(
    train_dataset, config["train_batch_size"], shuffle=True)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, config["val_batch_size"], shuffle=False)

# initialize device ('cuda', or 'cpu')
device = torch.device(config["device"])

# initialize device ('cuda', or 'cpu')
device = torch.device(config["device"])

# create model
model = Transformer(
    config["vocab_size"],
    config["vocab_size"],
    config["history_len"],
    config["response_len"],
    d_word_vec=config["embedding_dim"],
    d_model=config["model_dim"],
    d_inner=config["inner_dim"],
    n_layers=config["num_layers"],
    n_head=config["num_heads"],
    d_k=config["dim_k"],
    d_v=config["dim_v"],
    dropout=config["dropout"]
).to(device)


# optimizer class for updating the learning rate
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# create optimizer
optimizer = torch.optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    betas=(0.9, 0.98), eps=1e-09)

# create a sceduled optimizer object
optimizer = ScheduledOptim(
    optimizer, config["model_dim"], config["warmup_steps"])


def save_checkpoint(filename, model, optimizer):
    '''
    saves model into a state dict, along with its training statistics,
    and parameters
    :param model:
    :param optimizer:
    :return:
    '''
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, device):
    '''
    loads previous model
    :param filename: file name of model
    :param model: model that contains same parameters of the one you are loading
    :param optimizer:
    :return: loaded model, checkpoint
    '''
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


''' 
If you want to run a pretrained model, change the "old_model_dir" from None to the filename with the pretrained model
You must have the same vocab for the old model, so that is loaded as well
'''

'''if config["old_model_dir"] is not None:
    model, optimizer.optimizer = load_checkpoint(os.path.join(config["old_model_dir"], "model.bin"),
                                                 model, optimizer.optimizer, device)
    vocab.load_from_dict(os.path.join(config["old_model_dir"], "vocab.json"))'''


def greedy_output_example(model, val_dataset, device, vocab):
    '''output an example and the models prediction for that example'''
    random_index = random.randint(0, len(val_dataset))
    print('random_index', random_index)
    example = val_dataset[random_index]

    # prepare data
    h_seq, h_pos, h_seg, r_seq, r_pos = map(
        lambda x: torch.from_numpy(x).to(device).unsqueeze(0), example)

    # take out first token from target for some reason
    gold = r_seq[:, 1:]

    # forward
    pred = model(h_seq, h_pos, h_seg, r_seq, r_pos)
    output = torch.argmax(pred, dim=1)

    # get history text
    string = "history: "
    seg = -1
    for i, idx in enumerate(h_seg.squeeze()):
        if seg != idx.item():
            string += "\n"
            seg = idx.item()
        token = vocab.id2token[h_seq.squeeze()[i].item()]
        if token != '<blank>':
            string += "{} ".format(token)

    # get target text
    string += "\nTarget:\n"
    for idx in gold.squeeze():
        token = vocab.id2token[idx.item()]
        string += "{} ".format(token)

    # get prediction
    string += "\n\nPrediction:\n"
    for idx in output:
        token = vocab.id2token[idx.item()]
        string += "{} ".format(token)

    # print
    print("\n------------------------\n")
    print(string)
    print("\n------------------------\n")


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(transformer.Constants.PAD)
    # eq omputes element-wise equality
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(transformer.Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        # loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=transformer.Constants.PAD, reduction='mean')
    return loss


# forward
def forward(phase, batch, model, optimizer):
    h_seq, h_pos, h_seg, r_seq, r_pos = map(
        lambda x: x.to(device), batch)

    gold = r_seq[:, 1:]

    # forward
    if phase == "train":
        optimizer.zero_grad()
    pred = model(h_seq, h_pos, h_seg, r_seq, r_pos)

    return pred, gold


# backward
def backward(phase, pred, gold, config):
    # get loss
    loss, n_correct = cal_performance(pred, gold,
                                      smoothing=config["label_smoothing"])

    if phase == "train":
        # backward
        loss.backward()

        # update parameters, and learning rate
        optimizer.step_and_update_lr()

    return float(loss), n_correct


# initialize results, add config to them
results = dict()
results["config"] = config

# initialize lowest validation loss, use to save weights
lowest_loss = 999

# begin training
for i in range(config["num_epochs"]):
    start = time.time()
    epoch_metrics = dict()
    # output an example
    greedy_output_example(model, val_dataset, device, vocab)
    # run each phase per epoch
    for phase in ["train", "val"]:
        if phase == "train":
            # set model to training mode
            model.train()
            dataloader = data_loader_train
            batch_size = config["train_batch_size"]
        else:
            # set model to evaluation mode
            model.eval()
            dataloader = data_loader_val
            batch_size = config["val_batch_size"]

        # initialize metrics
        phase_metrics = dict()
        epoch_loss = list()
        average_epoch_loss = None
        n_word_total = 0
        n_correct = 0
        n_word_correct = 0
        for i, batch in enumerate(tqdm(dataloader, mininterval=2, desc=phase, leave=False)):
            # forward
            pred, gold = forward(phase, batch, model, optimizer)
            # backward
            loss, n_correct = backward(phase, pred, gold, config)

            # record loss
            epoch_loss.append(loss)
            average_epoch_loss = np.mean(epoch_loss)

            # get_accuracy
            non_pad_mask = gold.ne(transformer.Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        # record metrics
        phase_metrics["loss"] = average_epoch_loss
        phase_metrics["token_accuracy"] = n_word_correct / n_word_total

        # get perplexity
        perplexity = np.exp(average_epoch_loss)
        phase_metrics["perplexity"] = perplexity

        phase_metrics["time_taken"] = time.time() - start

        epoch_metrics[phase] = phase_metrics

        # save model if val loss is lower than any of the previous epochs
        if phase == "val":
            filename = config["output_dir"] + '/model.bin'  # + perplexity + '.bin'
            if i >= config["num_epochs"] - 5:
                filename = config["output_dir"] + '/model' + str(perplexity) + '.bin'
            if average_epoch_loss <= lowest_loss:
                save_checkpoint(filename, model, optimizer.optimizer)
                lowest_loss = average_epoch_loss
    epoch = i
    results["epoch_{}".format(epoch)] = epoch_metrics
