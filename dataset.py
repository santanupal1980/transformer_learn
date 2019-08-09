from collections import Counter
import numpy as np
import torch.utils.data
import csv
import json
from transformer import Constants


def read_a_file(_file):
    flist = list()
    with open(_file, "r", encoding='utf8') as fin:
        for line in fin:
            flist.append(line.split(" "))
        # print(len(list))
    return flist


def _read_file(source_file, target_file):
    source_sentences = read_a_file(source_file)
    target_sentences = read_a_file(target_file)



    return source_sentences, target_sentences


class Vocab(object):

    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self.nb_tokens = 0

        # vocab mapping
        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

    # updates the vocab with an example
    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = self.nb_tokens
                self.id2token[self.nb_tokens] = token
                self.nb_tokens += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    # prune the vocab that occur less than the min count
    def prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set([t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete -= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, self.nb_tokens))

    # load token2id from json file, useful when using pretrained model
    def load_from_dict(self, filename):
        with open(filename, 'r') as f:
            self.token2id = json.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

    # Save token2id to json file
    def save_to_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.token2id, f)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return 'Vocab: {} tokens'.format(self.nb_tokens)


class Dataset(torch.utils.data.Dataset):
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    SEP_WORD = '<s>'
    EOS_WORD = '</s>'
    CLS_WORD = '<cls>'

    def __init__(self, source_filename, target_filename, source_len=50, target_len=50, vocab=None, update_vocab=True):
        """
        Initialize the dataset.
        Get examples, and create/update vocab
        Examples:
            Source: <cls> hello ! <s> hi , how are you ? </s>
            Target: <cls> i am good , thank you ! </s>
        Args:
            filename: Filename of csv file with the data
            source_len: Maximum token length for the source. Will be
                pruned/padded to this length
            target_len: Maximum length for the target.
            vocab: Optional vocab object to use for this dataset
            update_vocab: Set to false to not update the vocab with the new
                examples
        """
        self.source, self.target= _read_file(source_filename, target_filename)


        self.source_len = source_len
        self.target_len = target_len

        if vocab is None:
            # Create new vocab object
            self.vocab = Vocab(special_tokens=[Constants.PAD_WORD,
                                               Constants.UNK_WORD,
                                               Constants.SEP_WORD,
                                               Constants.EOS_WORD,
                                               Constants.CLS_WORD])
        else:
            self.vocab = vocab

        # do not want to update vocab for running old model
        if update_vocab:
            self.vocab.add_documents(self.source)
            self.vocab.add_documents(self.target)

    def _process_source(self, source):
        """
        creates token encodings for the word embeddings, positional encodings,
        and segment encodings for the dialogue Source
        Examples:
            Source: <cls> hello ! <s> hi , how are you ? </s>
            self.source_len = 15
            s_seq = np.array([4, 34, 65, 2, 23, 44, 455, 97, 56, 10, 3, 0, 0, 0, 0])
            s_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0)]
            s_seg = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0)]
        Args:
            source: list of tokens in the source
        Returns:
            h_seq: token encodings for the source
            h_pos: positional encoding for the source
            h_seg: segment encoding for the source
        """
        source = source[-self.source_len + 1:] # cut from begining if > source_len
        source.append(Constants.EOS_WORD)

        needed_pads = self.source_len - len(source)
        if needed_pads > 0:
            source = source + [Constants.PAD_WORD] * needed_pads

        source = [
            self.vocab[token] if token in self.vocab else self.vocab[Constants.UNK_WORD]
            for token in source
        ]

        # create position embeddings, make zero if it is the pad token (0)
        s_pos = np.array([pos_i + 1 if w_i != 0 else 0
                          for pos_i, w_i in enumerate(source)])

        # create context embeddings
        seg = list()
        i = 1
        for j, token in enumerate(source):
            if token == self.vocab[Constants.PAD_WORD]:
                break
            seg.append(i)
            if token == self.vocab[Constants.SEP_WORD]:
                i += 1
        seg += [0] * needed_pads
        s_seg = np.array(seg, dtype=np.long)

        s_seq = np.array(source, dtype=np.long)

        return s_seq, s_pos, s_seg

    def _process_target(self, target):
        """
        creates token encodings for the word embeddings, and positional
            encodings for the target
        Examples:
            target:  <cls> i am good , thank you ! </s>
            self.target_len = 10
            r_seq = np.array([4, 43, 52, 77, 9, 65, 93, 5,  3, 0])
            r_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0,)]
        Args:
            target: list of tokens in the target
        Returns:
            r_seq: token encodings for the target
            r_pos: positional encoding for the target
        """
        target = target[:self.target_len - 1]
        target.append(Constants.EOS_WORD)
        # target.insert(0, DialogueDataset.CLS_WORD)

        needed_pads = self.target_len - len(target)
        if needed_pads > 0:
            target = target + [Constants.PAD_WORD] * needed_pads

        target = [
            self.vocab[token] if token in self.vocab else self.vocab[Constants.UNK_WORD]
            for token in target
        ]
        # create position embeddings
        t_pos = np.array([pos_i + 1 if w_i != 0 else 0
                          for pos_i, w_i in enumerate(target)])
        t_seq = np.array(target, dtype=np.long)
        return t_seq, t_pos

    def get_input_features(self, source):
        """ get features for chatbot """

        all_source = list()
        all_source.append(Constants.CLS_WORD)
        for line in source:
            all_source += list(line)
            all_source.append(Constants.SEP_WORD)
        h_seq, h_pos, h_seg = self._process_source(all_source[:-1])
        return torch.from_numpy(h_seq).unsqueeze(0), torch.from_numpy(h_pos).unsqueeze(0), torch.from_numpy(
            h_seg).unsqueeze(0)

    def __getitem__(self, index):
        """
            returns the features for an example in the dataset
        Args:
            index: index of example in dataset
        Returns:
            h_seq: token encodings for the Source
            h_pos: positional encoding for the Source
            h_seg: segment encoding for the Source
            r_seq: token encodings for the target
            r_pos: positional encoding for the target
        """
        s_seq, s_pos, s_seg = self._process_source(self.source[index])
        t_seq, t_pos = self._process_target(self.target[index])
        
        return s_seq, s_pos, s_seg, t_seq, t_pos

    def __len__(self):
        return len(self.source)