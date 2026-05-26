
# Dataset for adversarial competition.

Two datasets will be used for the competition:

* **DEV** dataset which is available here for development and experimenting.
* **TEST** dataset which will be kept secret until after the competition
  and will be used for final scoring.

# Dataset

This directory provides the metadata and scripts for preparing the ImageNet-compatible evaluation dataset used in our experiments.

The experiments use the development dataset from the NIPS 2017 Adversarial Attacks and Defenses Competition, which contains 1,000 images. The full image files are not included in this repository due to redistribution and file-size limitations.

## Files

- `dev_dataset.csv`: metadata file of the NIPS 2017 adversarial competition development dataset.
- `download_images.py`: script for downloading images from the URLs in `dev_dataset.csv`.
- `process.py`: script for converting downloaded images to the required sequential naming format.
- `images/`: directory for downloaded images. This directory is not included by default and will be created after downloading.

## Option 1: Download Images Using `dev_dataset.csv`

To download the evaluation images from the URLs provided in `dev_dataset.csv`, run:

```bash
python download_images.py --input_file=dev_dataset.csv --output_dir=images/
```

The downloaded images will be cropped according to the bounding boxes in `dev_dataset.csv`, resized, and saved in PNG format.

## Option 2: Download from Kaggle

Alternatively, users may download the NIPS 2017 adversarial competition dataset from Kaggle:

```text
https://www.kaggle.com/code/benhamner/adversarial-learning-challenges-getting-started
```

After downloading the images from Kaggle, please place them under:

```text
dataset/images/
```

If the downloaded image filenames do not follow the required sequential naming format, run:

```bash
python process.py
```

The `process.py` script renames all images in `dataset/images/` into the following format:

```text
1.png
2.png
3.png
...
```

The core renaming process is:

```python
images = os.listdir('./images/')
k = 1
for image in images:
    os.rename('./images/' + image, './images/' + '{0}.png'.format(k))
    k = k + 1
```

## Expected Directory Structure

After preparing the dataset, the directory should be organized as follows:

```text
dataset/
├── dev_dataset.csv
├── download_images.py
├── process.py
├── README.md
└── images/
    ├── 1.png
    ├── 2.png
    ├── 3.png
    └── ...
```

## Label Format

The dataset is labeled with ImageNet-compatible labels. The label file used by the attack and evaluation scripts should be placed in the root directory of the repository, for example:

```text
labels.txt
```

Please ensure that the image filenames are consistent with the label file used in the experiments.

## Notes

The full dataset images are not included in this repository. Users should download the images using either `download_images.py` or the Kaggle source described above.

Generated adversarial examples should not be placed in this directory. They should be saved to the output directory specified by the `--output_dir` argument in the attack script.