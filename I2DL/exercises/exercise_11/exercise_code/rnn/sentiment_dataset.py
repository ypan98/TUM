import os
import re
import pickle
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..util.download_utils import download_dataset


def download_data(data_dir='./data'):
    url = 'https://i2dl.vc.in.tum.de/static/data/SentimentData.zip'
    download_dataset(url, data_dir, 'SentimentData.zip')
    return os.path.join(data_dir, 'SentimentData')


def tokenize(text):
    return [s.lower() for s in re.split(r'\W+', text) if len(s) > 0]


def load_vocab(base_dir):
    vocab_file = os.path.join(base_dir, 'vocab.pkl')
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_sentiment_data(base_dir, vocab):
    train_file = os.path.join(base_dir, 'train_data.pkl')
    val_file = os.path.join(base_dir, 'val_data.pkl')
    test_file = os.path.join(base_dir, 'test_data.pkl')

    def load_data(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        unk = vocab['<unk>']
        result = []
        for text, label in data:
            tokens = tokenize(text)
            indices = [vocab.get(token, unk) for token in tokens]
            result.append((text, tokens, indices, label))
        return result

    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file)

    return train_data, val_data, test_data


def create_dummy_data(base_dir, sample_size=3, max_len=20, min_len=5):
    vocab = load_vocab(base_dir)
    train_data, _, _ = load_sentiment_data(base_dir, vocab)
    train_data1 = [
        (text, label)
        for text, tokens, _, label in train_data
        if min_len <= len(tokens) <= max_len and label == 1
    ]
    train_data0 = [
        (text, label)
        for text, tokens, _, label in train_data
        if min_len <= len(tokens) <= max_len and label == 0
    ]
    data = random.sample(train_data1, sample_size) + random.sample(train_data0, sample_size)
    return data


class SentimentDataset(Dataset):
    def __init__(self, data):
        """
        Inputs:
            data: list of tuples (raw_text, tokens, token_indices, label)
        """
        self.data = data
        self.data.sort(key=lambda x: len(x[1]), reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        """
        Inputs:
            i: an integer value to index data
        Outputs:
            data: A dictionary of {data, label}
        """
        _, _, indices, label = self.data[i]
        return {
            'data': torch.tensor(indices).long(),
            'label': torch.tensor(label).float()
        }


def collate(batch):
    """
        To be passed to DataLoader as the `collate_fn` argument
    """
    assert isinstance(batch, list)
    data = pad_sequence([b['data'] for b in batch])
    lengths = torch.tensor([len(b['data']) for b in batch])
    label = torch.stack([b['label'] for b in batch])
    return {
        'data': data,
        'label': label,
        'lengths': lengths
    }
