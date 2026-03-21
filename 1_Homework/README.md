# Lab: Retrieval as Training-free Classification on GTSRB

## Overview

This lab treats road sign classification as a **retrieval problem**, avoiding the need for fine-tuning by leveraging representations from massively pretrained backbones. The training set is used as a gallery of indexed image descriptors, and the test set as a set of query descriptors.

The pipeline consists of two stages:
1. **Retrieval evaluation** — rank test query by similarity to each galley images and measure mAP per class
2. **Nearest-Mean Classifier (NMC)** — classify test images by finding the nearest class mean in feature space


## Dataset

[GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/) — 43 classes of road signs.


## Project Structure

```
1_Homework/
├── main.py                  # Entry point
├── config/
│   └── models.yaml          # Model names and transform keys
├── src/
│   ├── dataset.py           # GTSRB dataloaders and several transforms
│   ├── feature_extractor.py # Generic feature extraction + head removal
│   ├── retrieval.py         # Similarity, ranking, AP, mAP
│   ├── classifier.py        # Nearest-Mean Classifier
│   ├── plot.py              # All plotting functions
│   └── utils.py             # Device selection, model path listing
├── models/                  # Saved feature tensors (.pt)
├── results/                 # mAP results per model (.pt)
├── classify/                # NMC accuracy results per model (.pt)
└── plots/                   # Generated plots
```



## Backbones Evaluated

Models were selected to cover a range of architectures, from classic CNNs to modern transformers:

 - `efficientnet_b0`
 - `efficientnet_v2_s`
 - `resnet50`       
 - `swin_t`
 - `vit_b_16`
 - `resnet18`

All models use `weights='DEFAULT'` (pretrained on ImageNet) with no fine-tuning.



## Results

### mAP (Retrieval)

| Model | mAP |
|-------|-----|
| `resnet50` |  |
| `resnet18` |  |
| `efficientnet_b0` |  |
| `efficientnet_v2_s` |  |
| `swin_t` |  |
| `vit_b_16` |  |


### NMC Accuracy

| Model | Accuracy |
|-------|----------|
| `resnet50` |  |
| `resnet18` |  |
| `efficientnet_b0` |  |
| `efficientnet_v2_s` |  |
| `swin_t` |  |
| `vit_b_16` |  |


---

## How to Run

```bash
python main.py
```

Models and transforms are configured in `config/models.yaml`:

```yaml
efficientnet_b0 : None 
efficientnet_v2_s: None
resnet50 : None       
swin_t: None
vit_b_16 : vit
resnet18 : None
```
The trasform string specifies which type of transform to apply to the model. If None standard transformation is applied.

Features tensors are cached in `models/` — if a model's features are already saved, extraction is skipped automatically.