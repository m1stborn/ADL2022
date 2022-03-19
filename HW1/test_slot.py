import csv
import json
import os.path
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
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

    # model = SlotClassifier(
    #     embeddings,
    #     hidden_size=64,
    #     dropout=0.1,
    #     bidirectional=True,
    #     num_class=9,
    #     num_layers=3
    # )
    # model.to(device=device)

    model = SlotClassifier(
        embeddings,
        hidden_size=512,
        dropout=0.1,
        bidirectional=True,
        num_class=9,
        num_layers=2
    )
    model.to(device=device)

    ckpt = load_checkpoint(args.ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        # Testing:
        test_ids = []
        test_pred = []
        test_token_lens = []
        for test_data in test_dataloader:
            inputs = test_data['tokens'].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 2)

            test_pred.append(predicted)
            test_ids = test_ids + test_data['id']
            test_token_lens = test_token_lens + test_data['lens']

        test_pred = torch.cat(test_pred).cpu().numpy()

        # print(test_pred[0])
        # print(test_pred[0][:token_lens[0]])
        # print([dataset.idx2label(tag) for tag in test_pred[0][:token_lens[0]]])

        # Write result to csv file
        with open(args.pred_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'tags'])

            for idx, tags_idx, l in zip(test_ids, test_pred, test_token_lens):
                tags = [dataset.idx2label(tag) for tag in tags_idx[:l]]
                writer.writerow((idx, ' '.join(tags)))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
        default="./data/slot/test.json"
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
        required=True,
        default="./ckpt/slot"
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
