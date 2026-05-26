# Dynamic Frequency-Guided Multi-Feature Attention Attack (DFG-MFAA)

Corresponding code to the paper "Dynamic Frequency-Guided Multi-Feature Attention for Transferable Adversarial Attacks"

DFG-MFAA improves the transferability of adversarial examples by combining multi-feature attention, residual feature response recalibration, and dynamic frequency-guided gradient modulation.

<img width="6297" height="4171" alt="final_2samples_compare" src="https://github.com/user-attachments/assets/00967f0a-7c39-4865-91a2-c793181ac72e" /># Dynamic Frequency-Guided Multi-Feature Attention Attack (DFG-MFAA)


# Requirements

- Python 3.9.15
- Keras 2.8.0
- TensorFlow 2.8.1
- NumPy 1.26.4
- Pillow 4.2.1
- SciPy 1.5.4
- absl-py 2.3.1

# Experiments

#### Introduction

- `DFG_MFAA.py` : the implementation of the proposed DFG-MFAA attack.

- `verify.py` : the code for evaluating generated adversarial examples on different models.

- `utils.py` : utility functions for model loading, image preprocessing, and checkpoint path configuration.

- `nets/` : TensorFlow-Slim network definitions used by the attack and evaluation scripts.

- `dataset/` : metadata and scripts for preparing the ImageNet-compatible evaluation dataset.

You should download the pretrained models from:

- https://github.com/tensorflow/models/tree/master/research/slim
- https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models

before running the code. Then place these model checkpoint files in `./models_tf`.

#### Dataset

The experiments use the ImageNet-compatible development dataset from the NIPS 2017 Adversarial Attacks and Defenses Competition, which contains 1,000 images.

The full image files are not included in this repository due to redistribution and file-size limitations. The dataset can be prepared using the files in `dataset/`.

You can download images using:

```bash
cd dataset
python download_images.py --input_file=dev_dataset.csv --output_dir=images/
```

Alternatively, the dataset can also be obtained from Kaggle:

```text
https://www.kaggle.com/code/benhamner/adversarial-learning-challenges-getting-started
```

If the downloaded images do not follow the required naming format, place them under `dataset/images/` and run:

```bash
python process.py
```

This will rename the images as:

```text
1.png
2.png
3.png
...
```

#### Example Usage

##### Generate adversarial examples:

- DFG-MFAA

```bash
python DFG_MFAA.py \
  --model_name inception_v3 \
  --attack_method MFA \
  --ens 30 \
  --probb 0.8 \
  --image_size 299 \
  --image_resize 330 \
  --output_dir ./adv/DFG_MFAA/
```

##### Evaluate the attack success rate

```bash
python verify.py \
  --ori_path ./dataset/images/ \
  --adv_path ./adv/DFG_MFAA/ \
  --output_file ./log.csv
```

# Notes

This repository provides a basic artifact for anonymous review. Full reproduction requires the evaluation dataset and pretrained model checkpoints, which are not included in this repository.

Generated adversarial examples will be saved to the directory specified by `--output_dir`.
