import json
import uuid
from argparse import ArgumentParser, Namespace
from typing import Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
)

from dataset import CtxSleDataset

CONTEXT = "context"
DEV = "valid"
SPLITS = [CONTEXT, DEV]
UID = str(uuid.uuid1())


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    # Model
    model = AutoModelForMultipleChoice.from_pretrained(args.ckpt)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text(encoding='utf-8')) for split, path in data_paths.items()}
    context_data = data[CONTEXT]

    datasets: Dict[str, ] = {
        split: CtxSleDataset(split_data, context_data, args.max_len, tokenizer)
        for split, split_data in data.items()
    }

    # Data
    eval_dataloader = DataLoader(
        datasets[DEV], shuffle=False, batch_size=args.batch_size
    )

    # Evaluation
    metric = load_metric("accuracy")
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        data = {k: v.to(device) for k, v in batch["tokenized_examples"].items()}
        with torch.no_grad():
            outputs = model(**data)
        predicted = outputs.logits.argmax(dim=-1)
        print(predicted)
        metric.add_batch(
            predictions=predicted,
            references=batch["label"],
        )
        if step > 10:
            break
    eval_metric = metric.compute()
    print(f"Valid Metric: {eval_metric}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        # required=True,
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/ctx_sle"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=384)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
