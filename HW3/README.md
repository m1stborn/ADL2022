# Title Generation

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
    <li><a href="#Slot-Tagging">Slot Tagging</a></li>
</details>

### Training
```shell
python train_title_gen.py
```

### Inference
```shell
bash download.sh
bash run.sh <test_file> <path to predicion>
```
### Model Performance

| Strategy        | rouge-1   | rouge-2   | rouge-L   |
|:----------------|-----------|-----------|-----------|
| greedy          | 24.82     | 9.437     | 22.26     |
| beams=2         | 25.78     | 10.25     | 23.12     |
| **beams=4**     | **26.12** | **10.62** | **23.47** |
| temperature=0.7 | 21.58     | 7.665     | 19.30     |
| temperature=0.9 | 17.66     | 5.823     | 15.88     |
| top 50          | 19.60     | 6.422     | 17.37     |
| top 30          | 20.55     | 6.893     | 18.25     |

Final generation strategy: beams = 4

###### tags: `NTU` `ADL` `2022`
