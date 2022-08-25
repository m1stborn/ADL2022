# Intent Classification and Slot Tagging

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
    <li><a href="#Preprocessing">Preprocessing</a></li>
    <li><a href="#Intent-Classification">Intent Classification</a></li>
    <li><a href="#Slot-Tagging">Slot Tagging</a></li>
</details>

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

# Intent Classification
### Training
```shell
python train_intent.py
```

### Inference
```shell
bash intent_cls.sh <test_file> <path to predicion>
```

### Model Performance
| Model | Accuracy |
|-------|----------|
| BiGRU | 0.8902   |


# Slot Tagging

### Training
```shell
python train_slot.py
```

### Inference
```shell
bash download.sh
bash slot_tag.sh <test_file> <path to predicion>
```
### Model Performance

| Model  | Accuracy |
|--------|----------|
| BiLSTM | 0.7973   |

###### tags: `NTU` `ADL` `2022`