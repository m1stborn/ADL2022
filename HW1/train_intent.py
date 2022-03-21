import os
import json
import uuid
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import spacy
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from utils import Vocab, save_checkpoint
from model import SeqClassifier

torch.manual_seed(1)
torch.cuda.manual_seed(1)

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
UID = str(uuid.uuid1())


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_dataloader = DataLoader(datasets[TRAIN], batch_size=64, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_dataloader = DataLoader(datasets[DEV], batch_size=64, shuffle=True, collate_fn=datasets[DEV].collate_fn)

    # Usage of dataloader
    # batch = next(iter(train_dataloader))
    # print(batch)
    # print(len(batch['text'][0]))
    # print(batch['text'])

    # Usage of vocab
    # print(datasets[TRAIN][0])
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(datasets[TRAIN][0]["text"])
    # tokenized = [token.text for token in doc]
    # print(vocab.encode(tokenized))

    embeddings = torch.load(args.cache_dir / "embeddings.pt")  # 6491 x 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = SeqClassifier(
        embeddings,
        hidden_size=64,
        dropout=0.1,
        bidirectional=True,
        num_class=150,
        num_layers=3
    )
    model.to(device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Usage of model
    # out = model(batch['text'].to(device))

    pre_val_acc = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0.0

        epoch_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)
        for i, data in epoch_pbar:
            inputs, labels = data['text'].to(device), data['intent'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Set Progress Bar
            epoch_pbar.set_description(f'Epoch[{epoch + 1}/{args.num_epoch}]')
            epoch_pbar.set_postfix(loss=running_loss / (i + 1))

        # Print Evaluation Statistic
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(valid_dataloader):
                inputs, labels = data['text'].to(device), data['intent'].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Valid Acc: {correct / total:.4f}")

        if pre_val_acc < (correct / total):
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'optim': optimizer.state_dict(),
                'uid': UID,
                'acc': (correct / total)
            }
            save_checkpoint(checkpoint,
                            os.path.join(args.ckpt_dir, "Intent-{}.pt".format(UID[:8])))
            pre_val_acc = correct / total
            print(f"Epoch {checkpoint['epoch']} saved!")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
