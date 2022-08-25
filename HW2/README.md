# Question Answering

A two stage pipeline for Question Answering. 
The first model is a Multiple Choice Model which will select a context that is relevant to the question out of four. 
The second model is a Question Answering Model which generate answer span from the choosing context.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
    <li><a href="#ContextSelection">Context Selection</a></li>
    <li><a href="#Question Answering">Question Answering</a></li>
</details>

# Context Selection
## Training
```shell
python train_ctx_sle.py
```

## Model Performance
| Model             | Accuracy |
|-------------------|----------|
| bert-base-chinese | 0.95     |


# Question Answering
## Training
```shell
python train_qa.py
```

## Inference
```shell
bash download.sh
bash run.sh <ctx_file.json> <path to predicion>
```

## Model Performance

| Model                   | EM   |
|-------------------------|------|
| chinese-roberta-wwm-ext | 0.81 |

###### tags: `NTU` `ADL` `2021`
