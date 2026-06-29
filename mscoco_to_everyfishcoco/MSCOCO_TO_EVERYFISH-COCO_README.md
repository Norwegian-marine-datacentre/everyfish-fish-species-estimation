# COCO to EVERYFISH-COCO Converter

A Python script that converts MS-COCO format annotations into EVERYFISH-COCO format ([Zhou _et al._,2026](https://doi.org/10.21335/NMDC-450800267)).

## Overview

This script takes:
- A single JSON file in **MS-COCO format** containing all annotations
- A **folder of images** (including those in subfolders)

And produces:
- **annotations/** - One JSON file per image containing image metadata and all annotations for that image
- **images/** - All images copied from source (preserving subfolder structure if present)
- **info/** - Metadata files (categories.json, licenses.json, info.json, metadata.json)

## Quick Start

### Basic Usage

```bash
python3 coco_to_per_image_format.py --coco_json <path_to_coco_json> --images <path_to_images_folder> --output <output_folder>
```

### Example

```bash
python3 coco_to_per_image_format.py --coco_json annotations.json --images ./images --output ./output
```

This will create:
```
output/
├── annotations/
│   ├── image1.json
│   ├── image2.json
│   └── ...
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── info/
    ├── categories.json
    ├── info.json
    ├── licenses.json
    └── metadata.json

```

## Configuration File

The converter reads `info_metadata.yml` to populate the following metadata JSON files in the `info/` output folder:

- `licenses`: list of license definitions
- `info`: dataset-level metadata
- `metadata`: per-image metadata defaults or records

Example `info_metadata.yml`:

```yaml
licenses:
  - id: "1"
    name: ""
    url: ""
  - id: "2"
    name: ""
    url: ""
  ...

info:
  year: ""
  version: ""
  description: ""
  contributor: ""
  url: ""
  date_created: ""
  sharing_availability: ""

metadata:
  gear_type: ""
  mesh_selectivity_device: ""
  targeted_fish: ""
  fish_region: "" # the acronym of the RFMO (Regional Fisheries Management Organization)
```

> **NOTE**: The `metadata.image_name` field is not defined here because each image record is generated per input image file.

## Output Format

### EVERYFISH-COCO

Each image gets a JSON file with the following structure as an example:

```json
{
  "image": {
    "license": 1,
    "file_name": "image1.png",
    "width": 640,
    "height": 480,
    "date_captured": "01/01/2026",
    "image_id": 1,
    "annotations_type": ["detection", "segmentation", "weight_estimatation", "length_estimation"]
  },
  "annotation": [
    {
      "id": 0,
      "image_id": 1,
      "category_id": 5,
      "bbox": [100, 50, 200, 150], #  [x, y, width, height] from top left corner
      "segmentation": [[100, 50, 300, 50, 300, 200, 100, 200]],
      "area": 30000, # in pixels
      "length": 25.5, # (in cm)
      "weight": 1.2, # (in kg)
      "occlusion_type": "partially visible" # int [0(fully visible),  1(partially visible), 2(immeasurable) ],
      "annotator_type": "human" # [‘human’, ‘machine’ or ‘unknown’] 
    },
    ...
  ]
}
```

### Info Folder Files

- **categories.json**: Array of category objects where each object includes:
  - `id`: category identifier as a string
  - `name`: species name
  - `FAO`: FAO species code  
  Example:
  ```json
  [
    {
      "id": "0",
      "name": "Thunnus obesus",
      "FAO": "BET"
    }
  ]
  ```

- **info.json**: Dataset-level metadata fields including:
  - `year`
  - `version`
  - `description`
  - `contributor`
  - `url`
  - `date_created`
  - `sharing_availability`  
  Example template:
  ```json
  {
    "year": "2026",
    "version": "1.0",
    "description": "EVERYFISH Dataset",
    "contributor": "EVERYFISH Project Partners",
    "url": "https://everyfish.eu/",
    "date_created": "01/01/2026",
    "sharing_availability": "Public"
  }
  ```

- **licenses.json**: Array of license objects describing usage terms, with fields:
  - `id`
  - `name`
  - `url`  
  Example template:
  ```json
  [
    {
      "id": "1",
      "name": "MIT License",
      "url": "https://opensource.org/licenses/MIT"
    }
  ]
  ```

- **metadata.json**: Array of per-image dataset metadata entries. Each object typically contains a record for one image with fields:
  - `gear_type`
  - `mesh_selectivity_device`
  - `targeted_fish`
  - `fish_region`
  - `image_name`  
  Example:
  ```json
  [
    {
      "gear_type": "DIV",
      "mesh_selectivity_device": "0_0_0",
      "targeted_fish": "CEP",
      "fish_region": "ICCAT",
      "image_name": "batch_02_55500.png"
    }
  ]
  ```
