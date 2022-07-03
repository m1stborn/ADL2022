import json
from dataclasses import dataclass

import nltk
from transformers import (
    PreTrainedTokenizerBase,
)


@dataclass
class PreprocessTitleGenTrain:
    """
    """
    tokenizer: PreTrainedTokenizerBase
    text_column: str = "article"
    summary_column: str = "highlights"
    prefix: str = ""
    padding: str = "max_length"
    ignore_pad_token_for_loss: bool = True
    max_source_length: int = 1024
    max_target_length: int = 64

    def __call__(self, examples):
        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)

        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        print(model_inputs)
        return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def save_log(data, filename: str):
    json_object = json.dumps(data, indent=4)

    with open(filename, "w") as outfile:
        outfile.write(json_object)
