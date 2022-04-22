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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default="./ckpt/qa/79352469/log.json",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("assets"))

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
