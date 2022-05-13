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
    e = list(range(1, 9))

    assert len(e) == len(log['train_loss'])

    my_dpi = 151
    plt.figure(1, figsize=(3840 / my_dpi, 2160 / my_dpi), dpi=my_dpi)

    # Plot loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(e, log['train_loss'], label='train loss', marker='o', ls='--')

    plt.title("Loss", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.savefig(os.path.join(args.out_dir, "loss_history.jpg"))

    # Plot Rouge

    fig = plt.figure(1, figsize=(3840 / my_dpi, 2160 / my_dpi), dpi=my_dpi)
    r1 = [r["rouge1"] for r in log["rouge"]]
    r2 = [r["rouge2"] for r in log["rouge"]]
    r_l = [r["rougeL"] for r in log["rouge"]]
    r_l_sum = [r["rougeLsum"] for r in log["rouge"]]

    # plt.yticks(np.arange(10, 20 + 1, 0.1))
    # plt.plot(e, r1, label='rouge1', marker='o', ls='--', alpha=0.5)
    # plt.plot(e, r2, label='rouge2', marker='o', ls='--', alpha=0.5)
    # plt.plot(e, r_l, label='rougeL', marker='o', ls='--', alpha=0.5)
    # plt.plot(e, r_l_sum, label='rougeLsum', marker='', ls='--', alpha=0.5)

    plt.plot(e, r1, label='rouge1', marker='o', ls='--')
    plt.plot(e, r2, label='rouge2', marker='o', ls='--')
    plt.plot(e, r_l, label='rougeL', marker='o', ls='--')
    plt.plot(e, r_l_sum, label='rougeLsum', marker='', ls='--')

    plt.title("Rouge Score", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Rouge Score", fontsize=15)


    plt.savefig(os.path.join(args.out_dir, "rouge_history.jpg"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default="./ckpt/c0bf55e7/log.json",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("assets"))

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    arg.out_dir.mkdir(parents=True, exist_ok=True)
    main(arg)
