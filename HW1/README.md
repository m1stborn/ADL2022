# Intent Classification and Slot Tagging

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
    <li><a href="#Preprocessing">Preprocessing</a></li>
    <li><a href="#Intent Classification">Intent Classification</a></li>
    <li><a href="#Slot Tagging">Slot Tagging</a></li>
</details>

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

# Intent Classification
## Training
Intent detection
```shell
python train_intent.py
```

## Inference
Intent detection
```shell
bash intent_cls.sh <test_file> <path to best_checkpoint.pt>
```

## Model Performance
| Model | Accuracy |
|-------|----------|
| BiGRU | 0.8902   |


# Slot Tagging

## Training
Slot tagging
```shell
python train_slot.py
```

## Inference
Slot tagging
```shell
bash slot_tag.sh <test_file> <path to best_checkpoint.pt>
```
## Model Performance

| Model  | Accuracy |
|--------|----------|
| BiLSTM | 0.7973   |

###### tags: `NTU` `ADL` `2021`