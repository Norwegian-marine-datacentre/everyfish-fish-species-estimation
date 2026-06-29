# Fish Species Estimation

## Description

A tool for automated fish species identification. This pipeline uses a ResNet model to identify and classify fish species from images. The model was trained and tested on images of fish on a conveyor belt.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Setup](#dataset-setup)
  - [Running Inference](#running-inference)
  - [Output](#output)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Installation

### System Requirements

* Python 3.12+
* OpenCV 4.13
* PyTorch 2.11.0
* Conda environment

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Usage

### Dataset Setup

:exclamation: **IMPORTANT**: This tool requires a pre-annotated dataset.   

Whether you use your own dataset or the samples provided below, the annotations need to be converted to EVERYFISH-COCO format ([Zhou _et al._,2026](https://doi.org/10.21335/NMDC-450800267)) prior to use of this tool. See [MSCOCO\_TO\_EVERYFISHCOCO\_README.md](mscoco_to_everyfishcoco/MSCOCO_TO_EVERYFISH-COCO_README.md) for detailed conversion instructions.

Two sample datasets are available for use:

- **AutoFish** ([Bengtson *et al.*, 2025](https://doi.org/10.48550/arXiv.2501.03767))  
Follow the instructions below to download the dataset:
```bash
sudo apt-get install git-lfs
git clone https://huggingface.co/datasets/vapaau/autofish
cd autofish
git lfs pull
```
- **FDWE** ([Sokolova _et al._, 2023](https://doi.org/10.1093/icesjms/fsad118))  
Download from the 4TU.Research Data repository: [Sokolova _et al._, 2023](https://doi.org/10.4121/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3.v3)

### Running Inference

### Step 1: Download Pre-trained Weights

Pre-trained ResNet weights for FDWE and AutoFish datasets are available [here](weights/).

#### Training Info

The model uses ResNet50 from PyTorch. Data was preprocessed with the default transforms from the model.

The optimal model checkpoint was selected based on its performance on the validation dataset, evaluating the following metrics:

* Validation Loss & Accuracy: To monitor overall model convergence.
* Precision: To minimize false positives (ensuring fish assigned to a species actually belong to it).
* Recall: To minimize false negatives (ensuring the model successfully identifies all available instances of a species).

The model was trained with the following parameters:

* **Batch size:** 64
* **Epochs:** 100
* **Dataset split:** 70% training, 20% validation, 10% test

### Step 2: Crop Extraction and Ground Truth Generation

The `extract_crops.py` script extracts crops from images based on annotations and organizes them into class-specific folders. It also generates ground truth files for later comparison with inference results.

**Configuration:** Use `config/extract_crops_cfg.yml`

```bash
python3 extract_crops.py --config config/extract_crops_cfg.yml
```

**Crop extraction and ground truth generation output**

The output will have the following folder structure

```
output_dir/
├── crops/
│   ├── 1_Yellowfin_Tuna/
│   │   ├── 123_image001_1.png
│   │   └── ...
│   └── ...
├── ground_truth.csv
└──  summary_ground_truth.csv
```

The script can also optionally split the extracted crops into `train`, `val`, and `test` folders based on the provided percentages. For example: 
```bash
python3 extract_crops.py --config config/extract_crops_cfg.yml --split-ds --train-percent 70 --val-percent 20
```
By default the split is `70%` train, `20%` validation, and `10%` test.

```
output_dir/
├── train/
│   ├── 1_Yellowfin_Tuna/
│   │   ├── 123_image001_1.png
│   │   └── ...
│   └── ...
├── val/
│   ├── 1_Yellowfin_Tuna/
│   │   ├── 456_image005_2.png
│   │   └── ...
│   └── ...
├── test/
│   ├── 1_Yellowfin_Tuna/
│   │   ├── 789_image010_3.png
│   │   └── ...
│   └── ...
├── ground_truth.csv
├── summary_ground_truth.csv
├── train.txt
├── val.txt
└── test.txt
```

* `ground_truth.csv` contains detailed ground truth annotations per crop containing: crop_filename, crop_path, original_image, annotation_id, image_id, category_id, category_name, bbox coordinates
* `summary_ground_truth.csv` contains summary ground truth statistics with: class_id, class_name, count (crops per class), class_directory
* `train.txt`, `val.txt`, `test.txt` - contains a lists of crop filenames placed in each split folder (if applicable)

### Step 3: Fish Identification

The `fish_identification.py` script performs classification on pre-extracted crop images.

**Configuration:** Use `config/fish_identification_cfg.yml`

```bash
python3 fish_identification.py --config config/fish_identification_cfg.yml
```

The script expects crops under `crops_dir` and can recursively process individual or nested folders such as `train`, `val`, and `test`.

**Notes:**

* `ground_truth_csv` should point to the detailed `ground_truth.csv` produced by `extract_crops.py`
* The comparison logic now matches predictions to ground truth at the crop level using `crop_path` (with filename fallback)

**Fish identification output**

The fish_identification pipeline generates the following .csv files in the output directory:

1. **`class_stats.csv`** - Classification statistics per class:

   * `class_id` - Numeric class identifier
   * `class_name` - Name of the fish species
   * `count` - Number of detections for this class
   * `percentage` - Percentage of total detections
2. **`comparison.csv`** - (Optional) Ground truth comparison if `ground_truth_csv` is provided in config:

   * `class_id` - Class identifier
   * `class_name` - Species name
   * `gt_count` - Ground truth count
   * `pred_count` - Predicted count
   * `gt_percentage` - Ground truth percentage
   * `pred_percentage` - Predicted percentage
   * `absolute_error` - Absolute difference in count
   * `relative_error` - Relative error percentage
3. **`predictions.csv`** - Individual prediction results:

   * `image_id` - Source image identifier
   * `class_id` - Predicted class ID
   * `class_name` - Predicted species name
   * `confidence` - Confidence score (0.0-1.0)

**Terminal Output Summary:**

* Total detections and per-class breakdown
* If ground truth is available:

  * Mean Absolute Error (MAE)
  * Mean Absolute Percentage Error (MAPE)
  * Number of matched classes
  * Per-class error metrics

## Contributing
* AZTI

## License


## Acknowledgement

![European Union logo and funding acknowledgement](assets/EN_FundedbytheEU_RGB_POS.png)
We acknowledge funding from the EVERYFISH project. This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101059892.
