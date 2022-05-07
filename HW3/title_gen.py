from typing import List

import datasets
import jsonlines
from datasets import load_dataset

logger = datasets.logging.get_logger(__name__)


class TitleGenConfig(datasets.BuilderConfig):
    """BuilderConfig for Title Gen dataset."""
    jsonl_files: List[str] = None
    split_names: List[str] = None

    def __init__(self, **kwargs):
        """BuilderConfig for Title Gen dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TitleGenConfig, self).__init__(**kwargs)


class TitleGenDataset(datasets.GeneratorBasedBuilder):
    """Title Gen dataset: The ADL@NTU Homework 3 Dataset. Version 1.0."""

    BUILDER_CONFIGS = [
        TitleGenConfig(
            name="Title Gen",
            version=datasets.Version("1.0.0", ""),
            description="Jsonlines file",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Title Gen dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "highlights": datasets.Value("string")
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        split_name = {
            "train": datasets.Split.TRAIN,
            "eval": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        return [
            datasets.SplitGenerator(
                name=split_name[split],
                gen_kwargs={
                    "jsonlines_file": jsonl_file,
                    "split": split,
                }
            ) for jsonl_file, split in zip(self.config.jsonl_files, self.config.split_names)
        ]

    def _generate_examples(self, jsonlines_file, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", jsonlines_file)
        is_test = split == "test"

        key = 0
        with open(jsonlines_file, encoding="utf-8") as f:
            jsonl_object = jsonlines.Reader(f)

            if not is_test:
                for article in jsonl_object:
                    yield key, {
                        "id": article["id"],
                        "article": article["maintext"],
                        "highlights": article["title"]
                    }
                    key += 1
            else:
                for article in jsonl_object:
                    yield key, {
                        "id": article["id"],
                        "article": article["maintext"],
                        "highlights": ""
                    }
                    key += 1


if __name__ == '__main__':
    # TitleGenDateset example
    dev_files = ["./data/train.jsonl", "./data/public.jsonl", "./data/sample_test.jsonl"]
    splits = ["train", "eval", "test"]

    raw_dataset = load_dataset("title_gen.py", name="Title Gen", cache_dir="./cache",
                               jsonl_files=dev_files, split_names=splits)

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"]
    test_dataset = raw_dataset["test"]

    print(train_dataset[0])
    print(eval_dataset[0])
    print(test_dataset[0])
