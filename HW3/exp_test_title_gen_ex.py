import csv
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    AutoConfig,
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
)

from utils import PreprocessTitleGenTrain


accelerator = Accelerator()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(1)


def main(args):
    # Config Tokenizer and Model
    config = AutoConfig.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dev_files = ["./data/public.jsonl"]
    splits = ["eval"]
    raw_datasets = load_dataset("title_gen.py", name="Title Gen",
                                jsonl_files=dev_files, split_names=splits)

    # TODO: make utils function to keep id column
    all_id = [features["id"] for features in raw_datasets["validation"]]
    column_names = raw_datasets["validation"].column_names

    preprocess_function = PreprocessTitleGenTrain(tokenizer=tokenizer)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["validation"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    gen_kwargs_set = [
        # {
        #     "max_length": args.max_target_length,
        #     "num_beams": args.num_beams,
        # },
        # {
        #     "max_length": args.max_target_length,
        #     "num_beams": 2,
        #     "num_return_sequences": 1,
        #     "early_stopping": True,
        # },
        # {
        #     "max_length": args.max_target_length,
        #     "num_beams": 4,
        #     "num_return_sequences": 1,
        #     "early_stopping": True,
        # },
        # {
        #     "max_length": args.max_target_length,
        #     "do_sample": True,
        #     "top_k": 0,
        #     "temperature": 0.7,
        # },
        # {
        #     "max_length": args.max_target_length,
        #     "do_sample": True,
        #     "top_k": 50,
        # }
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 0,
            "temperature": 0.9,
        },
        {
            "max_length": args.max_target_length,
            "do_sample": True,
            "top_k": 100,
        }
    ]
    output_filenames = [
        # "./assets/result-origin.jsonl",
        # "./assets/result-beams_2.jsonl",
        # "./assets/result-beams_4.jsonl",
        # "./assets/result-temp_0.7.jsonl",
        # "./assets/result-top_50.jsonl",
        "./assets/result-temp_0.9.jsonl",
        "./assets/result-top_100.jsonl",
    ]

    # Testing
    for gen_kwargs, filename in zip(gen_kwargs_set, output_filenames):
        all_pred = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                all_pred += decoded_pred

                if step == 1 and args.dev:
                    break

        print(all_pred[:3], all_id[:3])

        if not args.dev:
            # Write result to csv file
            with open(filename, 'w') as file:
                for idx, title in zip(all_id, all_pred):
                    json.dump({"title": title, "id": idx}, file)
                    file.write('\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # file path
    parser.add_argument("--test_file", type=Path, default="./data/public.jsonl")
    parser.add_argument("--pred_file", type=Path, default="result-dev.jsonl")

    # ckpt folder
    parser.add_argument("--ckpt", type=Path, default="./ckpt/07d92fbf")

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # summary generation
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)

    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arges = parse_args()
    main(arges)
