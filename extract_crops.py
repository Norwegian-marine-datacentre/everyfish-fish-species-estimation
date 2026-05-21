"""
This script extracts crops from images based on annotations and organizes them
into class-specific folders. It also generates a ground truth file for later
comparison with inference results.

Features:
    - Extracts crops using bounding boxes (detection) or segmentation masks
    - For segmentation: creates transparent background outside mask area
    - Organizes crops into class-specific subfolders
    - Generates ground truth CSV files for evaluation
"""

import logging
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from utils.logging import setup_logging
from utils.common import CategoryManager, AnnotationLoader, ImageProcessor

import cv2

logger = setup_logging("INFO")


class CropExtractionConfig:
    """
    Configuration manager for crop extraction.

    Loads and validates configuration from YAML file.
    """

    REQUIRED_KEYS = [
        'annotations_dir', 'images_dir', 'categories_file',
        'annotation_type', 'output_dir'
    ]
    VALID_ANNOTATION_TYPES = ['detection', 'segmentation']

    def __init__(self, config_path: str):
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

        # Validate required keys
        missing_keys = [key for key in self.REQUIRED_KEYS if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Validate annotation type
        if config_dict['annotation_type'] not in self.VALID_ANNOTATION_TYPES:
            raise ValueError(
                f"Invalid annotation_type. Must be one of {self.VALID_ANNOTATION_TYPES}"
            )

        self.annotations_dir = Path(config_dict['annotations_dir'])
        self.images_dir = Path(config_dict['images_dir'])
        self.categories_file = Path(config_dict['categories_file'])
        self.annotation_type = config_dict['annotation_type']
        self.output_dir = Path(config_dict['output_dir'])

        self.dataset_name = config_dict.get('dataset_name', 'Unknown Dataset')

        self._validate_paths()
        logger.info(f"Configuration loaded successfully. Annotation type: {self.annotation_type}")

    def _validate_paths(self):
        """Validate that all required paths exist."""
        paths_to_check = [
            ('annotations_dir', self.annotations_dir),
            ('images_dir', self.images_dir),
            ('categories_file', self.categories_file),
        ]

        for name, path in paths_to_check:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")


class CropExtractor:
    """
    Extracts crops from images and organizes them by class.

    """

    def __init__(self, config: CropExtractionConfig, category_manager: CategoryManager):
        """
        Initialize crop extractor.

        :param config: CropExtractionConfig instance
        :param category_manager: CategoryManager instance
        """
        self.config = config
        self.category_manager = category_manager
        self.image_processor = ImageProcessor(config.images_dir)

        self.crops_dir = config.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Create class-specific subdirectories
        self.class_dirs = {}
        for cat_id, cat_name in self.category_manager.categories.items():
            # Sanitize class name for folder name
            safe_name = cat_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            class_dir = self.crops_dir / f"{cat_id}_{safe_name}"
            class_dir.mkdir(exist_ok=True)
            self.class_dirs[cat_id] = class_dir

        # Ground truth data collection
        self.ground_truth_data = []

        logger.info(f"Created output directory: {config.output_dir}")
        logger.info(f"Created {len(self.class_dirs)} class directories")

    def extract_all_crops(self):
        """
        Extract crops from all annotation files.

        :return:Dictionary with extraction statistics
        """
        logger.info("="*70)
        logger.info("STARTING CROP EXTRACTION")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Annotation Type: {self.config.annotation_type}")
        logger.info("="*70)

        # Get all annotation files
        annotation_files = list(self.config.annotations_dir.glob('*.json'))
        logger.info(f"Found {len(annotation_files)} annotation files")

        total_crops = 0
        processed_images = 0
        error_count = 0

        # Process each annotation file
        for annotation_file in tqdm(annotation_files, desc="Processing annotation files"):
            try:
                loader = AnnotationLoader(annotation_file, logger)
                image_info = loader.get_image_metadata()

                try:
                    crops_extracted = self._process_image(loader, image_info)
                    total_crops += crops_extracted
                    processed_images += 1
                except Exception as e:
                    logger.warning(f"Error processing image {image_info['id']}: {e}")
                    error_count += 1

            except Exception as e:
                logger.error(f"Error processing annotation file {annotation_file}: {e}")
                error_count += 1

        # Save ground truth file
        self._save_ground_truth()

        logger.info("="*70)
        logger.info("CROP EXTRACTION COMPLETE")
        logger.info(f"Successfully processed: {processed_images} images")
        logger.info(f"Total crops extracted: {total_crops}")
        logger.info(f"Errors encountered: {error_count}")
        logger.info(f"Ground truth saved to: {self.config.output_dir}")
        logger.info("="*70)

        return {
            'total_crops': total_crops,
            'processed_images': processed_images,
            'error_count': error_count,
            'classes': len(self.class_dirs)
        }

    def _process_image(self, loader: AnnotationLoader, image_info: dict) -> int:
        """
        Process a single image and extract all its crops.

        :param loader: AnnotationLoader instance
        :param image_info: Image metadata dictionary

        :return: Number of crops extracted from this image
        """
        image_id = image_info['image_id']
        image_filename = image_info['file_name']

        image = self.image_processor.load_image(image_filename)

        # Get annotations
        annotations = loader.get_image_annotations(image_id)

        if not annotations:
            logger.debug(f"No annotations found for image {image_id}")
            return 0

        crops_extracted = 0

        # Process each annotation
        for annotation in annotations:
            try:
                # Extract crop based on annotation type
                if self.config.annotation_type == 'detection':
                    if 'bbox' not in annotation or annotation.get('bbox') is None:
                        # try to extract box from mask
                        if 'segmentation' in annotation and annotation.get('segmentation') is not None:
                            segmentation = annotation['segmentation']
                            crop = self.image_processor.crop_by_segmentation(image, segmentation[0], make_transparent=False)
                            
                            coords = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                            x_min = coords[:, 0].min()
                            y_min = coords[:, 1].min()
                            x_max = coords[:, 0].max()
                            y_max = coords[:, 1].max()
                            bbox_coords = [x_min, y_min, x_max - x_min, y_max - y_min]
                        else:
                            logger.warning(f"No bbox in annotation for image {image_id}")
                            continue

                    else:
                        crop = self.image_processor.crop_by_bbox(image, annotation['bbox'])
                        bbox_coords = annotation['bbox']

                elif self.config.annotation_type == 'segmentation':
                    if 'segmentation' not in annotation:
                        logger.warning(f"No segmentation in annotation for image {image_id}")
                        continue

                    segmentation = annotation['segmentation']
                    crop = self.image_processor.crop_by_segmentation(image, segmentation[0])
                    
                    # Calculate bbox from segmentation for ground truth
                    coords = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                    x_min = coords[:, 0].min()
                    y_min = coords[:, 1].min()
                    x_max = coords[:, 0].max()
                    y_max = coords[:, 1].max()
                    bbox_coords = [x_min, y_min, x_max - x_min, y_max - y_min]

                # Skip empty crops
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    logger.debug(f"Empty crop for annotation {annotation.get('id', 'unknown')}")
                    continue

                # Save crop to appropriate class directory
                self._save_crop(crop, annotation, image_filename, bbox_coords)
                crops_extracted += 1

            except Exception as e:
                logger.debug(f"Error processing annotation {annotation.get('id', 'unknown')}: {e}")

        return crops_extracted

    def _save_crop(self, crop: np.ndarray, annotation: dict, image_filename: str,
                   bbox_coords: list[float]):
        """
        Save crop image to appropriate class directory and record ground truth.

        :param crop: Cropped image as numpy array
        :param annotation: Annotation dictionary
        :param image_filename: Original image filename
        :param bbox_coords: Bounding box coordinates [x, y, w, h]
        """
        category_id = annotation['category_id']
        image_id = annotation.get('image_id', 'unknown')
        annotation_id = annotation.get('id', 'unknown')

        # Get class directory
        if category_id not in self.class_dirs:
            logger.warning(f"Unknown category ID {category_id}, skipping crop")
            return

        class_dir = self.class_dirs[category_id]

        image_basename = Path(image_filename).stem
        ext = 'png'
        crop_filename = f"{image_id}_{annotation_id}_{image_basename}_{category_id}.{ext}"
        crop_path = class_dir / crop_filename

        success = cv2.imwrite(str(crop_path), crop)
        if not success:
            logger.warning(f"Failed to save crop: {crop_path}")
            return

        # Record ground truth information
        ground_truth_entry = {
            'crop_filename': crop_filename,
            'crop_path': str(crop_path.relative_to(self.config.output_dir)),
            'original_image': image_filename,
            'annotation_id': annotation_id,
            'image_id': annotation['image_id'],
            'category_id': category_id,
            'category_name': self.category_manager.get_name(category_id),
            'bbox_x': bbox_coords[0],
            'bbox_y': bbox_coords[1],
            'bbox_width': bbox_coords[2],
            'bbox_height': bbox_coords[3],
        }

        self.ground_truth_data.append(ground_truth_entry)

    def _save_ground_truth(self):
        """Save ground truth data to CSV file."""

        if not self.ground_truth_data:
            logger.warning("No ground truth data to save")
            return

        logger.info(f"Saving ground truth data for {len(self.ground_truth_data)} crops")

        df = pd.DataFrame(self.ground_truth_data)

        csv_path = self.config.output_dir / "ground_truth.csv"
        df.to_csv(csv_path, index=False)

        # Create summary sheet
        summary_path = self.config.output_dir / "summary_ground_truth.csv"
        summary_data = []
        for cat_id in sorted(self.category_manager.categories.keys()):
            cat_name = self.category_manager.get_name(cat_id)
            cat_crops = df[df['category_id'] == cat_id]
            summary_data.append({
                'class_id': cat_id,
                'class_name': cat_name,
                'count': len(cat_crops),
                'class_directory': str(self.class_dirs[cat_id].relative_to(self.config.output_dir))
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"Ground truth saved to: {csv_path}")
        logger.info(f"Summary saved to: {summary_path}")


    def split_dataset(self, train_percent: float, val_percent: float, test_percent: float):
        """
        Split the dataset into training, validation, and test sets.

        :param train_percent: Percentage for training set
        :param val_percent: Percentage for validation set
        :param test_percent: Percentage for test set
        """
        if not self.ground_truth_data:
            logger.warning("No ground truth data available for splitting")
            return

        # Get unique image filenames
        unique_images = set()
        for entry in self.ground_truth_data:
            unique_images.add(entry['original_image'])

        image_list = list(unique_images)
        np.random.shuffle(image_list)

        total_images = len(image_list)
        train_count = int(total_images * train_percent / 100)
        val_count = int(total_images * val_percent / 100)

        # Split the list
        train_images = image_list[:train_count]
        val_images = image_list[train_count:train_count + val_count]
        test_images = image_list[train_count + val_count:]

        # Create mapping from image to split
        image_to_split = {}
        for img in train_images:
            image_to_split[img] = 'train'
        for img in val_images:
            image_to_split[img] = 'val'
        for img in test_images:
            image_to_split[img] = 'test'

        # Create split directories and class subfolders
        for split in ['train', 'val', 'test']:
            split_dir = self.config.output_dir / split
            split_dir.mkdir(exist_ok=True)
            for cat_id, cat_name in self.category_manager.categories.items():
                safe_name = cat_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                class_dir = split_dir / f"{cat_id}_{safe_name}"
                class_dir.mkdir(exist_ok=True)

        # Move crops to appropriate split folders
        for entry in self.ground_truth_data:
            original_image = entry['original_image']
            split = image_to_split[original_image]
            category_id = entry['category_id']
            cat_name = self.category_manager.get_name(category_id)
            safe_name = cat_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            new_class_dir = self.config.output_dir / split / f"{category_id}_{safe_name}"
            old_path = self.config.output_dir / entry['crop_path']
            new_path = new_class_dir / old_path.name
            shutil.move(str(old_path), str(new_path))
            entry['crop_path'] = str(new_path.relative_to(self.config.output_dir))
            entry['split'] = split

        # Collect crops per split
        split_crops = {'train': [], 'val': [], 'test': []}
        for entry in self.ground_truth_data:
            split = entry['split']
            crop_filename = Path(entry['crop_path']).name  # just the filename
            split_crops[split].append(crop_filename)

        # Write crop txt files
        for split_name, crops in split_crops.items():
            filename = f"{split_name}.txt"
            filepath = self.config.output_dir / filename
            with open(filepath, 'w') as f:
                for crop in crops:
                    f.write(f"{crop}\n")
            logger.info(f"Saved {len(crops)} crops to {filepath}")

        logger.info(f"Dataset split complete: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        # Delete crops directory after moving
        shutil.rmtree(self.crops_dir)
        logger.info("Crops moved to train/, val/, test/ folders with class subfolders")


def main():
    parser = argparse.ArgumentParser(
        description='Extract crops from images and create ground truth files',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--config', type=str, default='config/extract_crops_cfg.yml', help='Path to YAML configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--split-ds', action='store_true', default=False, help='Split the dataset into train, val, and test sets')
    parser.add_argument('--train-percent', type=float, default=70.0, help='Percentage of images for training set (default: 70)')
    parser.add_argument('--val-percent', type=float, default=20.0, help='Percentage of images for validation set (default: 20)')

    args = parser.parse_args()

    # Compute test percentage
    test_percent = 100.0 - args.train_percent - args.val_percent
    if test_percent < 0:
        raise ValueError(f"Train and validation percentages must sum to less than 100. Current sum: {args.train_percent + args.val_percent}")

    # Update logger level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = CropExtractionConfig(args.config)

        # Load categories
        category_manager = CategoryManager(config.categories_file, logger)

        # Create extractor and run
        extractor = CropExtractor(config, category_manager)
        stats = extractor.extract_all_crops()

        # Split dataset
        if args.split_ds:
            extractor.split_dataset(args.train_percent, args.val_percent, test_percent)

        # Save updated ground truth with split info
        extractor._save_ground_truth()

        # Print summary
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY")
        print("="*70)
        print(f"Total crops extracted: {stats['total_crops']}")
        print(f"Images processed: {stats['processed_images']}")
        print(f"Classes: {stats['classes']}")
        print(f"Errors: {stats['error_count']}")
        print(f"Output directory: {config.output_dir}")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Crop extraction failed: {e}")
        raise


if __name__ == '__main__':
    main()
