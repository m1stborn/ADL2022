from typing import List, Dict

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
            self,
            data: List[Dict],
            vocab: Vocab,
            label_mapping: Dict[str, int],
            max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        ids = [sample['id'] for sample in samples]

        tokens = [sample['text'].split(' ') for sample in samples]
        encoded_tokens = self.vocab.encode_batch(tokens, to_len=self.max_len)

        # Only Train and Eval data has labels
        try:
            labels = [self.label2idx(sample['intent']) for sample in samples]
        except KeyError:
            labels = []

        return {'text': torch.tensor(encoded_tokens, dtype=torch.int),
                'intent': torch.tensor(labels, dtype=torch.long),
                'id': ids}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SlotTagDataset(Dataset):
    def __len__(self) -> int:
        return len(self.data)

    def __init__(
            self,
            data: List[Dict],
            vocab: Vocab,
            label_mapping: Dict[str, int],
            max_len: int,
            pad_tag_idx: int = -1,
            mask: bool = False,
            mask_token: str = "[MASK]",
            mask_prob: float = 0.05,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.pad_tag_idx = pad_tag_idx
        self.mask = mask
        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.mask_multiple_length = 1

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def __getitem__(self, index) -> Dict:
        if self.mask:
            # Randomly mask the sequence
            sz = len(self.data[index]['tokens'])
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz / float(self.mask_multiple_length)
                + np.random.rand()
            )
            # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask[mask_idc] = True

            self.data[index]['tokens'][mask.tolist() == True] = self.mask_token
            instance = self.data[index]
        else:
            instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        ids = [sample['id'] for sample in samples]

        tokens = [sample['tokens'] for sample in samples]

        encoded_tokens = self.vocab.encode_batch(tokens, to_len=self.max_len)
        lens = [len(sample['tokens']) for sample in samples]
        # Only Train and Eval data has labels
        try:
            tags = [[self.label2idx(tag) for tag in sample['tags']]
                    for sample in samples]
        except KeyError:
            tags = []
        padded_tags = pad_to_len(tags, self.max_len, self.pad_tag_idx)

        return {'tokens': torch.tensor(encoded_tokens, dtype=torch.int),
                'tags': torch.tensor(padded_tags, dtype=torch.long),
                'lens': lens,
                'id': ids}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
