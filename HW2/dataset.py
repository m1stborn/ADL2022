from itertools import chain
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy


class CtxSleDataset(Dataset):
    def __init__(
            self,
            data: List[Dict],
            ctx_data: List[str],
            max_len: int,
            tokenizer: PreTrainedTokenizerBase,
    ):
        self.data = data
        self.ctx_data = ctx_data
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        ctx_id = instance["paragraphs"]
        contexts = {f"ending{i}": self.ctx_data[ctx_id[i]]
                    for i in range(4)}

        # second_sentences = [v for k, v in contexts.items()]
        first_sentences = [instance['question'] for i in range(4)]
        second_sentences = [contexts[f"ending{i}"] for i in range(4)]

        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=self.max_len,
            padding="max_length",
            return_tensors='pt',
            truncation=True,
        )

        label = instance["paragraphs"].index(instance['relevant'])

        return {
            'id': index,
            'question': instance['question'],
            # 'label': torch.tensor(label, dtype=torch.int64),
            'label': label,
            'relevant': instance['relevant'],
            **contexts,
            'tokenized_examples': tokenized_examples,
            'paragraphs': instance["paragraphs"],
        }


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop('label') for feature in features]
        ids = [feature.pop('id') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["tokenized_examples"]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature["tokenized_examples"].items()} for i in range(num_choices)] for feature in
            features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            verbose=False,
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["ids"] = ids

        return batch


# if __name__ == "__main__":
#     import json
#     from pathlib import Path
#     ctx_path = Path('./data/context.json')
#     train_path = Path('./data/train.json')
#
#     from transformers import (
#         AutoTokenizer,
#     )
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
#
#     ctx_data = json.loads(ctx_path.read_text(encoding='utf-8'))
#     train_data = json.loads(train_path.read_text(encoding='utf-8'))
#     train_dataset = CtxSleDataset(train_data, ctx_data, 384, tokenizer)
#
#     data_collator = DataCollatorForMultipleChoice(
#         tokenizer, pad_to_multiple_of=8
#     )
#
#     from torch.utils.data import DataLoader
#     train_dataloader = DataLoader(
#         train_dataset, shuffle=False, collate_fn=data_collator, batch_size=8
#     )
#
#     instance = train_dataset.data[0]
#     ctx_idx = instance['relevant']
#     label = instance['paragraphs'].index(ctx_idx)
#
#     ctx = train_dataset.ctx_data[ctx_idx]
#     question = instance['question']
#
#     # Check real instance
#     print(instance, label)
#     print(tokenizer(ctx)["input_ids"][:100], len(tokenizer(ctx)["input_ids"]))
#     print(tokenizer(question)["input_ids"], len(tokenizer(question)["input_ids"]))
#
#     # Check batch with collator_fn
#     batch = next(iter(train_dataloader))
#     print(batch["ids"])
#     print(batch["input_ids"][0][label])
