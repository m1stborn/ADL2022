## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```

## Slot tagging
```shell
python train_slot.py
```

## Intent Classification Result

| Model | Accuracy |
|-------|----------|
| BiGRU | 0.8902   |

## Slot Tagging Result

| Model  | Accuracy |
|--------|----------|
| BiLSTM | 0.7973   |
