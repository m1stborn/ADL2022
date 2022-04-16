import csv
import json
import uuid
from itertools import chain
from argparse import ArgumentParser, Namespace
from typing import Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from dataset import CtxSleDataset, DataCollatorForMultipleChoice
from utils_qa import postprocess_qa_predictions
from utils import create_and_fill_np_array, data_collator

# Pipeline:
#   1. Context Selection: Select one related contex out of 4 candidate.
#   2. Question Answering: Select answer span from the contex.

CONTEXT = "context"
TEST = "test"
SPLITS = [CONTEXT, TEST]
UID = str(uuid.uuid1())

accelerator = Accelerator()


def main(args):
    # 1. Context Selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ctx_sle_ckpt)

    # Model
    ctx_sle_model = AutoModelForMultipleChoice.from_pretrained(args.ctx_sle_ckpt)
    ctx_sle_model.to(device)
    ctx_sle_model.resize_token_embeddings(len(tokenizer))

    # # TODO: Refactor test file name
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text(encoding='utf-8')) for split, path in data_paths.items()}
    context_data = data[CONTEXT]
    #
    # datasets: Dict[str, ] = {
    #     split: CtxSleDataset(split_data, context_data, args.max_len, tokenizer, mode="test")
    #     for split, split_data in data.items()
    # }

    # data_collator = DataCollatorForMultipleChoice(
    #     tokenizer, pad_to_multiple_of=8
    # )
    # Data
    # TODO: modify batch size
    # test_dataloader = DataLoader(
    #     datasets[TEST], shuffle=False, collate_fn=data_collator, batch_size=32
    # )

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
            second_sentences,
            first_sentences,
            max_length=args.max_len,
            padding=padding,
            truncation="only_first",
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs["labels"] = labels

        return tokenized_inputs
    raw_test_dataset = load_dataset("ctx_sle.py", name="test", cache_dir="./cache2",
                                    question_file="./data/test.json", context_file="./data/context.json")

    test_dataset = raw_test_dataset["test"]
    with accelerator.main_process_first():
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
        )
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=8)

    # Testing
    all_ids = []
    all_paragraphs = []
    all_pred = []

    for step, batch in enumerate(test_dataloader):
        ids = batch.pop("ids")
        paragraphs = batch.pop("paragraphs")
        _ = batch.pop("labels")
        data = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = ctx_sle_model(**data)
        predicted = outputs.logits.argmax(dim=-1)

        all_ids += ids
        all_paragraphs += paragraphs
        all_pred.append(predicted)


    all_pred = torch.cat(all_pred).cpu().numpy()
    selected_context = {instance_id: paragraphs[contex_idx]
                        for instance_id, contex_idx, paragraphs in zip(all_ids, all_pred, all_paragraphs)}
    print(selected_context, len(selected_context))

    # 2. Question Answering
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.qa_ckpt)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_ckpt)

    raw_test_dataset = load_dataset("qa.py", name="test", cache_dir="./cache2",
                                    question_file="./data/test.json", context_file="./data/context.json")

    column_names = raw_test_dataset["test"].column_names
    question_column_name = "question"
    context_column_name = "context"

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_test_context(examples):
        # Fill the context according to the CtxSle model prediction
        examples[context_column_name] = [context_data[selected_context[idx]]
                                         for idx in examples["id"]]
        # examples[context_column_name] = [context_data[0]
        #                                  for example_id in examples["id"]]
        return examples

    def prepare_test_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    # Add the predicted context to test example
    with accelerator.main_process_first():
        test_example = raw_test_dataset["test"].map(
            prepare_test_context,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )
    print("Test Example:", test_example[0])

    with accelerator.main_process_first():
        test_dataset = test_example.map(
            prepare_test_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )
    print("Test Dataset:", test_dataset[0])

    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    qa_data_collator = default_data_collator
    test_dataloader = DataLoader(
        test_dataset_for_model, shuffle=False, collate_fn=qa_data_collator, batch_size=8
    )
    model, test_dataloader = accelerator.prepare(
        qa_model, test_dataloader
    )

    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # necessary to pad predictions and labels for being gathered
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    # prediction = post_processing_function(eval_example, eval_dataset, outputs_numpy)
    predictions = postprocess_qa_predictions(
        examples=test_example,
        features=test_dataset,
        predictions=outputs_numpy,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=None,
        prefix="eval",
    )

    # Write result to csv file
    with open(args.pred_file, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'answer'])

        for example_id, answer in predictions.items():
            writer.writerow((example_id, answer))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/test.json"
    )
    parser.add_argument(
        "--ctx_sle_ckpt",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/ctx_sle"
    )
    parser.add_argument(
        "--qa_ckpt",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/qa"
    )
    parser.add_argument("--pred_file", type=Path, default="result.csv")
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

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arges = parse_args()
    main(arges)
