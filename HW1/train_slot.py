import os
import json
import uuid
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
import spacy
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SlotTagDataset
from utils import Vocab, save_checkpoint
from model import SlotClassifier

torch.manual_seed(1)
torch.cuda.manual_seed(1)

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
UID = str(uuid.uuid1())


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, ] = {
        split: SlotTagDataset(split_data, vocab, tag2idx, args.max_len, args.pad_tag_idx)
        for split, split_data in data.items()
    }

    train_dataloader = DataLoader(datasets[TRAIN], batch_size=128, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_dataloader = DataLoader(datasets[DEV], batch_size=128, shuffle=True, collate_fn=datasets[DEV].collate_fn)

    # Usage of dataloader
    batch = next(iter(train_dataloader))

    embeddings = torch.load(args.cache_dir / "embeddings.pt")  # 6491 x 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = SlotClassifier(
        embeddings,
        hidden_size=512,
        dropout=0.1,
        bidirectional=True,
        num_class=9,
        num_layers=2
    )
    model.to(device=device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.pad_tag_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Usage of model
    # out = model(batch['tokens'].to(device))
    # print(out.size())
    # _, predicted = torch.max(out.data, 2)
    # print(predicted[0])
    # print(batch['tags'][0])
    # print(out.size())

    pre_val_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0.0

        epoch_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)
        for i, data in epoch_pbar:
            inputs, labels = data['tokens'].to(device), data['tags'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Set Progress Bar
            epoch_pbar.set_description(f'Epoch[{epoch + 1}/{args.num_epoch}]')
            epoch_pbar.set_postfix(loss=running_loss / (i + 1))

        # Print Evaluation Statistic
        model.eval()
        with torch.no_grad():
            correct, total, valid_loss = 0, 0, 0.0

            for i, data in enumerate(valid_dataloader):
                inputs, labels = data['tokens'].to(device), data['tags'].to(device)
                token_lens = data['lens']

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 2)

                loss = criterion(outputs.transpose(1, 2), labels)
                valid_loss += loss.item()

                total += labels.size(0)
                correct += batch_tp(predicted, labels, token_lens)

            print(f"Valid Acc: {correct / total:.4f} Valid Loss: {valid_loss}")
        if pre_val_acc < (correct / total):
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'optim': optimizer.state_dict(),
                'uid': UID,
                'acc': (correct / total)
            }
            save_checkpoint(checkpoint,
                            os.path.join(args.ckpt_dir, "Slot-{}.pt".format(UID[:8])))
            pre_val_acc = correct / total
            print(f"Epoch {checkpoint['epoch']} saved!")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=35)
    parser.add_argument("--pad_tag_idx", type=int, default=-1)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=30)

    args = parser.parse_args()
    return args


def batch_tp(preds: torch.Tensor, targets: torch.Tensor, lens: List[int]) -> int:
    pred_np, target_np = preds.cpu().numpy(), targets.cpu().numpy()
    count = 0

    for pred, target, l in zip(pred_np, target_np, lens):
        # print(pred[:l], target[:l], l)
        count += (pred[:l] == target[:l]).all()

    # return number of true positive in a batch
    return count


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
