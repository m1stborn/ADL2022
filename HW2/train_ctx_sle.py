import json
import math
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from itertools import chain

import torch
from accelerate import Accelerator
from datasets import load_metric, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AutoConfig,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
    default_data_collator,
)

# from dataset import CtxSleDataset, DataCollatorForMultipleChoice
from ctx_sle import CtxSleDataset
from utils import data_collator

CONTEXT = "context"
TRAIN = "train"
DEV = "valid"
SPLITS = [CONTEXT, TRAIN, DEV]
UID = str(uuid.uuid1())


def main(args):
    set_seed(1)
    accelerator = Accelerator()

    # data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # data = {split: json.loads(path.read_text(encoding='utf-8')) for split, path in data_paths.items()}
    # context_data = data[CONTEXT]
    # # TODO: slow tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    #
    # datasets: Dict[str, ] = {
    #     split: CtxSleDataset(split_data, context_data, args.max_len, tokenizer)
    #     for split, split_data in data.items()
    # }

    # print(datasets[TRAIN][0])
    raw_train_dataset = load_dataset("ctx_sle.py", name="train", cache_dir="./cache2",
                                     question_file="./data/train.json", context_file="./data/context.json")

    raw_valid_dataset = load_dataset("ctx_sle.py", name="eval", cache_dir="./cache2",
                                     question_file="./data/valid.json", context_file="./data/context.json")

    ending_names = [f"ending{i}" for i in range(4)]
    padding = "max_length"

    # (Context|Question)
    def preprocess_function(examples):
        batch_size = len(examples["question"])
        # Question
        first_sentences = [[context] * 4 for context in examples["question"]]
        # Context
        second_sentences = [
            [examples[end][i] for end in ending_names] for i in range(batch_size)
        ]

        labels = examples["label"]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            # first_sentences,
            # second_sentences,
            second_sentences,
            first_sentences,
            max_length=args.max_len,
            padding=padding,
            truncation="only_first",
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    train_dataset = raw_train_dataset["train"]
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
        )
    print(train_dataset[0])
    eval_dataset = raw_valid_dataset["validation"]
    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
        )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    # batch = next(iter(train_dataloader))
    # print(batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epoch * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    # Metrics
    metric = load_metric("accuracy")

    # Train
    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0.0
        epoch_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100)

        for step, batch in epoch_pbar:
            _ = batch.pop("ids")
            _ = batch.pop("paragraphs")

            data = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**data)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()

                # Set Progress Bar
                epoch_pbar.set_description(f'Epoch[{epoch + 1}/{args.num_epoch}]')
                epoch_pbar.set_postfix(loss=running_loss / (step + 1))

        # Save Model
        tokenizer.save_pretrained(args.ckpt_dir)
        model.save_pretrained(args.ckpt_dir)

        # Evaluation
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            _ = batch.pop("ids")
            _ = batch.pop("paragraphs")
            data = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**data)
            predicted = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predicted,
                references=batch["labels"],
            )
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
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/ctx_sle/",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=6,
        help="Num worker for preprocessing"
    )

    # data
    parser.add_argument("--max_len", type=int, default=384)

    # model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="bert-base-chinese",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # training
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
