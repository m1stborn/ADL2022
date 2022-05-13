import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
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
    dev_files = [args.test_file]
    splits = ["test"]
    raw_datasets = load_dataset("title_gen.py", name="Title Gen",
                                jsonl_files=dev_files, split_names=splits)

    # TODO: make utils function to keep id column
    all_id = [features["id"] for features in raw_datasets["test"]]

    preprocess_function = PreprocessTitleGenTrain(tokenizer=tokenizer)
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=['id', 'article', 'highlights'],
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # gen_kwargs = {
    #     "max_length": args.max_target_length,
    #     "num_beams": args.num_beams,
    # }
    gen_kwargs = {
         "max_length": args.max_target_length,
         "num_beams": 4,
         "num_return_sequences": 1,
         "early_stopping": True,
    }

    # Testing
    all_pred = []
    epoch_pbar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=100)
    for step, batch in epoch_pbar:
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

            if step == 10 and args.dev:
                break

    print(all_pred[:3], all_id[:3])

    # Write result to csv file
    with open(args.pred_file, 'w') as file:
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
    # TODO: modify for submission => 4
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
