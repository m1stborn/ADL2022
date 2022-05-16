import json
import csv
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import default_data_collator, Trainer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
from datasets import load_metric, load_dataset
import argparse

tokenizer = AutoTokenizer.from_pretrained('t5-small')
tokenizer.pad_token = tokenizer.eos_token
max_input_length = 60
max_target_length = 30


def preprocess_function(examples):
    inputs = [ex for ex in examples['inputs']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding='max_length',
        add_special_tokens=True,
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding='max_length',
            add_special_tokens=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class OTTersDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data):
        tokenizer.pad_token = tokenizer.eos_token
        self.encodings = tokenizer(data['inputs'], padding=True, truncation=True)
        with tokenizer.as_target_tokenizer():
            self.targets = tokenizer(data['targets'], padding=True, truncation=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.targets['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def read_data(data_dir):
    splits = ['train', 'dev', 'test']
    datasets = {}
    for split in splits:
        directory = os.path.join(data_dir, split)
        datasets[split] = load_dataset(directory, data_files=['text.csv'])
        if split != 'test':
            datasets[split] = datasets[split].map(
                preprocess_function,
                batched=True,
                remove_columns=['inputs', 'target'],
            )['train']
        else:
            datasets[split] = datasets[split]['train']
    return datasets['train'], datasets['dev'], datasets['test']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default='OTTers/data', type=str, help="OTTers dir"
    )
    parser.add_argument(
        "--domain", default='in_domain', type=str, help="domain"
    )
    parser.add_argument(
        "--model_name_or_path", default='t5-small', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--output_dir", default='runs/finetune', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--max_epoch", default=10, type=int, help="total number of epoch"
    )
    parser.add_argument(
        "--train_bsize", default=16, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_bsize", default=16, type=int, help="evaluation batch size"
    )
    parser.add_argument(
        "--patience", default=3, type=int, help="early stopping patience"
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss", default=3, type=int, help="early stopping patience"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Dataset
    print('reading dataset')
    dataset_dir = os.path.join(args.dataset_root, args.domain)
    train_dataset, eval_dataset, test_dataset = read_data(dataset_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    metric = load_metric("sacrebleu")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print(preds[0])
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # Train model
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.eval_bsize,
        label_smoothing_factor=0.1,
        eval_accumulation_steps=10,
        # weight_decay=0.01,               # strength of weight decay
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # data_collator=default_data_collator,
        # compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # test
    model.to('cpu')
    inputs = tokenizer(test_dataset['inputs'], return_tensors="pt", padding=True)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))
    result = metric.compute(predictions=[predictions], references=test_dataset['target'])
    result = {"bleu": result["score"]}
    print(result)

    trainer.save_model(args.output_dir)
