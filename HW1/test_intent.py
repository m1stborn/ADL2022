import csv
import json
import os.path
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, load_checkpoint

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

    # Usage of dataloader
    # batch = next(iter(test_dataloader))
    # print(batch)
    # print(len(batch['text'][0]))
    # print(batch['text'].size())
    # print(batch['id'])
    # print(batch['intent'])

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SeqClassifier(
        embeddings,
        hidden_size=64,
        dropout=0.1,
        bidirectional=True,
        num_class=150,
        num_layers=3
        # args.hidden_size,
        # args.num_layers,
        # args.dropout,
        # args.bidirectional,
        # dataset.num_classes,
    )
    model.to(device)

    # ckpt = torch.load(args.ckpt_path)

    ckpt = load_checkpoint(args.ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        # Testing:
        test_ids = []
        test_pred = []
        for test_data in test_dataloader:
            inputs = test_data['text'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_pred.append(predicted)
            test_ids = test_ids + test_data['id']
        test_pred = torch.cat(test_pred).cpu().numpy()

        # Write result to csv file
        # pred_filename = os.path.join(args.pred_file, f"Intent-{ckpt['uid'][:8]}-result.csv")
        with open(args.pred_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'intent'])

            for idx, intent_idx in zip(test_ids, test_pred):
                writer.writerow((idx, dataset.idx2label(intent_idx)))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default="./ckpt/intent"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
