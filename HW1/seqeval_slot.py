import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from seqeval.scheme import IOB2
from seqeval.metrics import classification_report

from dataset import SlotTagDataset
from model import SlotClassifier

from utils import Vocab, load_checkpoint

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, intent2idx, args.max_len, args.pad_tag_idx)
    test_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlotClassifier(
        embeddings,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=9,
        num_layers=args.num_layers,
    )
    model.to(device=device)

    ckpt = load_checkpoint(args.ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        # Eval:
        test_ids = []
        test_pred = []
        test_token_lens = []
        test_tags = []

        total, correct = 0, 0

        for test_data in test_dataloader:
            inputs, labels = test_data['tokens'].to(device), test_data['tags'].to(device)
            token_lens = test_data['lens']

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 2)

            test_pred.append(predicted)
            test_tags.append(labels)
            test_ids = test_ids + test_data['id']
            test_token_lens = test_token_lens + test_data['lens']

            correct_token, l = batch_token_tp(predicted, labels, token_lens)
            correct += correct_token
            total += l
            break

        test_pred = torch.cat(test_pred).cpu().numpy()
        test_tags = torch.cat(test_tags).cpu().numpy()

        y_true = []
        y_pred = []
        for real_tags, tags, l in zip(test_tags, test_pred, test_token_lens):
            y_pred.append([dataset.idx2label(tag) for tag in tags[:l]])
            y_true.append([dataset.idx2label(tag) for tag in real_tags[:l]])

        print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
        print(f"Token Acc: {correct / total}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
        default="./data/slot/eval.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pad_tag_idx", type=int, default=-1)

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


def batch_token_tp(preds: torch.Tensor, targets: torch.Tensor, lens: List[int]) -> (int, int):
    pred_np, target_np = preds.cpu().numpy(), targets.cpu().numpy()
    count = 0
    token_count = 0

    for pred, target, l in zip(pred_np, target_np, lens):
        count += (pred[:l] == target[:l]).sum()
        token_count += l

    return count, token_count


if __name__ == "__main__":
    args = parse_args()
    main(args)
