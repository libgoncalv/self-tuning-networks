import os
import torch
import numpy as np
import random

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, corruption):
        self.dictionary = Dictionary()
        self.corruption = corruption
        self.train = self.corrupted_tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.corrupted_tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def corrupted_tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            lines = f.readlines()
        f = lines

        # Add words to the dictionary
        tokens = 0
        for i, line in enumerate(f):
            words = line.split()
            if np.random.random_sample()<=self.corruption:
                random.shuffle(words)
                f[i] = " ".join(words)
            words.append('<eos>')
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids
