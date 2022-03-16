from typing import List, Dict

import spacy
import torch
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
        self.nlp = spacy.load("en_core_web_sm")

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

        tokens = [[token.text for token in self.nlp(sample['text'])] for sample in samples]
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
    def __init__(
            self,
            data: List[Dict],
            vocab: Vocab,
            label_mapping: Dict[str, int],
            max_len: int,
            pad_tag_idx: -1,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.pad_tag_idx = pad_tag_idx

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

        tokens = [sample['tokens'] for sample in samples]
        encoded_tokens = self.vocab.encode_batch(tokens, to_len=self.max_len)
        lens = [len(sample['tokens']) for sample in samples]
        # Only Train and Eval data has labels
        try:
            # labels = [self.label2idx(sample['intent']) for sample in samples]
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

