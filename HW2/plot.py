import json
import os.path

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser, Namespace


def main(args):
    log = json.loads(args.log.read_text())

    # Plot
    plt.style.use('ggplot')
    e = list(range(0, 9560, 1000))
    e.append(9560)

    assert len(e) == len(log['train_loss'])

    log2 = json.loads(args.no_pretrain_log.read_text())
    e2 = [x + i*9560 for i in range(5) for x in e]
    assert len(e2) == len(log2['train_loss'])

    my_dpi = 151
    plt.figure(1, figsize=(3840 / my_dpi, 2160 / my_dpi), dpi=my_dpi)

    # Plot loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, log['train_loss'], label='train loss', marker='o', ls='--')

    plt.title("Loss", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.savefig(os.path.join(args.out_dir, "loss_history.jpg"))

    # Plot Exact match
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, log['valid_em'], label='valid EM', marker='o', ls='--', color='blue')

    plt.title("Exact match", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Exact match", fontsize=15)

    plt.savefig(os.path.join(args.out_dir, "em_history.jpg"))


    # Plot loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, log['train_loss'], label='pretrained', marker='o', ls='--')
    plt.plot(e2, log2['train_loss'], label='not pretrained', marker='o', ls='--')

    plt.title("Training Loss", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.savefig(os.path.join(args.out_dir, "loss_history_compare.jpg"))

    # Plot Exact match
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, log['valid_em'], label='pretrained', marker='o', ls='--')
    plt.plot(e2, log2['valid_em'], label='not pretrained', marker='o', ls='--')

    plt.title("Exact match", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Iteration", fontsize=15)
    plt.ylabel("Exact match", fontsize=15)

    plt.savefig(os.path.join(args.out_dir, "em_history_compare.jpg"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default="./ckpt/qa/79352469/log.json",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("assets"))

    parser.add_argument(
        "--no_pretrain_log",
        type=Path,
        default="./ckpt/qa/00329162/log.json",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
