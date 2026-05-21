"""
Fish Species Identification Inference Pipeline

This module implements a complete inference pipeline for fish species identification.
It processes images with EVERYFISH-COCO format annotations and a trained ResNet model to 
classify fish instances and generate detailed statistics.

It supports detection (bounding box), segmentation (mask), and individual crops as inputs.
The script exports classification results to various CSV files with the relevant statistics.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from utils.logging import setup_logging
from utils.common import CategoryManager, AnnotationLoader, ImageProcessor

import cv2
import torch
import torchvision.models as models
from torchvision.transforms import v2


logger = setup_logging("INFO")

class InferenceConfig:
    """
    Configuration manager for the inference pipeline.
    Loads and validates configuration from YAML file.
    
    Attributes:
        dataset_name (str): Name of the dataset
        weights_path (str): Path to trained model weights
        annotations_dir (Optional[Path]): Directory with annotation JSON files
        images_dir (Optional[Path]): Directory with image files
        crops_dir (Optional[Path]): Directory with pre-extracted crop images
        categories_file (str): Path to categories.json
        output_dir (str): Output directory for results CSV files
        batch_size (int): Batch size for inference
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        :param config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.annotation_type = None
        
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        missing_required = [
            key for key in ('dataset_name', 'weights_path', 'categories_file', 'output_dir')
            if key not in config_dict
        ]
        if missing_required:
            raise ValueError(f"Missing required config keys: {missing_required}")
        
        self.dataset_name = config_dict['dataset_name']
        self.weights_path = config_dict['weights_path']
        self.categories_file = Path(config_dict['categories_file'])
        self.output_dir = Path(config_dict['output_dir'])
        self.batch_size = config_dict.get('batch_size', 1)  # Default to 1 for single inference
        self.ground_truth_csv = Path(config_dict['ground_truth_csv']) if config_dict.get('ground_truth_csv') else None
        self.images_dir = Path(config_dict['images_dir']) if config_dict.get('images_dir') else None
        self.annotations_dir = Path(config_dict['annotations_dir']) if config_dict.get('annotations_dir') else None
        self.crops_dir = Path(config_dict['crops_dir']) if config_dict.get('crops_dir') else None
        
        self._validate_paths()
        logger.info(
            f"Configuration loaded successfully. annotation_type={self.annotation_type}"
        )
    
    def _validate_paths(self):
        """Validate that all required paths exist."""
        required_paths = [
            ('weights_path', self.weights_path),
            ('categories_file', self.categories_file),
        ]
        
        required_paths.append(('crops_dir', self.crops_dir))
        
        for name, path in required_paths:
            if path is None:
                raise ValueError(f"Missing required config path: {name}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")

class ResNetInferenceModel:
    """
    Wrapper for ResNet50 model inference.
    
    Handles model loading, preprocessing, and inference on fish image crops (both bounding box and segmenation mask)
    """
    
    def __init__(self, weights_path: str,
                 num_classes: int, 
                 device: str = 'cpu'):
        """
        Initialize and load ResNet model.
        
        :param weights_path: Path to saved model weights
        :param num_classes: Number of output classes
        :param device: Device to use ('cpu', 'cuda')
        """
        
        logger.info(f"Loading ResNet model from {weights_path}")
        
        self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.inference_transforms = models.ResNet50_Weights.DEFAULT.transforms()

        # Assign number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])

        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Preprocessing transforms
        logger.info("Using default preprocessing transforms")
        self.preprocess = v2.Compose([
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # ToTensor
        ])
    
    def predict(self, crop: np.ndarray | list[np.ndarray]) -> tuple[int, float] | list[tuple[int, float]]:
        """
        Perform inference on an image crop or batch of crops.
        
        :param crop: Image crop as numpy array (BGR format) or list of crops
            
        :return: tuple of (predicted_class_id, confidence_score) for single crop,
                 or list of tuples for batch
        """
        is_batch = isinstance(crop, list)
        crops = [crop] if not is_batch else crop
        
        # Validate crops
        for i, c in enumerate(crops):
            if c.size == 0 or c.shape[0] == 0 or c.shape[1] == 0:
                raise ValueError(f"Crop {i} is too small or invalid")
        
        try:
            # Convert BGR to RGB and preprocess all crops
            preprocessed_crops = []
            for c in crops:
                # Convert BGR to RGB for preprocessing
                crop_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                crop_rgb = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(crop_rgb)
                # Preprocess
                crop_rgb = self.inference_transforms(crop_rgb)
                preprocessed_crops.append(crop_rgb)
            
            # Stack into batch
            batch_tensor = torch.stack(preprocessed_crops).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
            
            predicted_classes = predicted_classes.cpu().tolist()
            confidences = confidences.cpu().tolist()
            
            results = list(zip(predicted_classes, confidences))
            
            return results[0] if not is_batch else results
        
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            raise

class ResultsManager:
    """
    Analyzes inference results, generates statistics, and
    compares against ground truth data.
    
    Attributes:
    - Tracking predictions per class
    - Calculating output statistics and percentages
    - Exporting results to CSV format
    - Printing summaries with ground truth comparison
    """
    
    def __init__(self, category_manager: CategoryManager, 
                 ground_truth_csv: Optional[Path] = None):
        """
        Initialize results manager.
        
        :param category_manager: CategoryManager instance for class name lookup
        :param ground_truth_csv: Path to ground truth CSV file for comparison
        """

        self.category_manager = category_manager
        self.predictions = []
        self.class_counts = defaultdict(int)
        self.expected_crop_paths = set()
        
        self.ground_truth_data = None
        self.ground_truth_df = None
        self.ground_truth_lookup = {}
        self.ground_truth_by_filename = {}
        self.ground_truth_detailed = False
        self.ground_truth_summary = {}
        
        if ground_truth_csv and ground_truth_csv.exists():
            self.ground_truth_data = self._load_ground_truth(ground_truth_csv)
            logger.info("Ground truth data loaded for comparison")
    
    def _normalize_path(self, path: str) -> str:
        normalized = os.path.normpath(path)
        return normalized.replace('\\', '/')

    def register_crop_path(self, crop_path: str):
        """Register a crop path for ground truth scope comparison."""
        self.expected_crop_paths.add(self._normalize_path(crop_path))

    def _load_ground_truth(self, ground_truth_csv: Path) -> dict[int, dict]:
        """
        Load ground truth data from a CSV file.
        
        Supports two formats:
          - Detailed ground_truth.csv with per-crop annotations
          - Summary summary_ground_truth.csv with class counts
        
        :param ground_truth_csv: Path to CSV file
        :return: Parsed ground truth mapping
        """
        if not ground_truth_csv.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")
        
        try:
            df = pd.read_csv(ground_truth_csv)
            
            if 'crop_path' in df.columns and 'category_id' in df.columns:
                self.ground_truth_detailed = True
                self.ground_truth_df = df
                for _, row in df.iterrows():
                    crop_path = str(row.get('crop_path') or row.get('crop_filename') or '')
                    if not crop_path or pd.isna(crop_path):
                        continue
                    crop_key = self._normalize_path(crop_path)
                    entry = {
                        'category_id': int(row['category_id']),
                        'category_name': str(row.get('category_name', '')),
                        'crop_filename': str(row.get('crop_filename', '')),
                        'crop_path': crop_key
                    }
                    self.ground_truth_lookup[crop_key] = entry
                    if entry['crop_filename']:
                        self.ground_truth_by_filename[entry['crop_filename']] = entry
                logger.debug(f"Loaded detailed ground truth for {len(self.ground_truth_lookup)} crops")
            elif 'class_id' in df.columns:
                for _, row in df.iterrows():
                    class_id = int(row['class_id'])
                    self.ground_truth_summary[class_id] = {
                        'name': str(row.get('class_name', '')),
                        'count': int(row['count']),
                        'percentage': float(row['percentage']) if 'percentage' in df.columns else 0.0
                    }
                logger.debug(f"Loaded summary ground truth for {len(self.ground_truth_summary)} classes")
            else:
                raise ValueError(
                    'Unsupported ground truth CSV format. Expected detailed ground_truth.csv or summary_ground_truth.csv'
                )
            logger.info(f"Loaded ground truth from {ground_truth_csv}")
        except Exception as e:
            logger.error(f"Error loading ground truth CSV: {e}")
            raise
        
        return self.ground_truth_summary if not self.ground_truth_detailed else self.ground_truth_lookup

    def _find_ground_truth(self, crop_path: str, crop_filename: str) -> Optional[dict]:
        """Find the ground truth entry for a crop by path or filename."""
        if not self.ground_truth_detailed:
            return None
        normalized_path = self._normalize_path(crop_path)
        if normalized_path in self.ground_truth_lookup:
            return self.ground_truth_lookup[normalized_path]
        if crop_filename in self.ground_truth_by_filename:
            return self.ground_truth_by_filename[crop_filename]
        return None
    
    def add_prediction(self, image_id: str, class_id: int, confidence: float, crop_path: Optional[str] = None):
        """
        Record a single prediction.
        
        :param image_id: ID of the image this prediction is from
        :param class_id: Predicted class ID
        :param confidence: Confidence score [0, 1]
        :param crop_path: Relative crop path used for ground truth lookup
        """
        prediction = {
            'image_id': image_id,
            'class_id': class_id,
            'class_name': self.category_manager.get_name(class_id),
            'confidence': confidence
        }
        if crop_path is not None:
            prediction['crop_path'] = self._normalize_path(crop_path)
        self.predictions.append(prediction)
        self.class_counts[class_id] += 1
    
    def get_statistics(self) -> dict:
        """
        Compute statistics from predictions.
        
        :return: Dictionary with aggregated statistics
        """
        total_detections = len(self.predictions)
        
        if total_detections == 0:
            return {'total_detections': 0, 'classes': {}}
        
        stats = {
            'total_detections': total_detections,
            'classes': {}
        }
        
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            percentage = (count / total_detections) * 100
            
            stats['classes'][class_id] = {
                'name': self.category_manager.get_name(class_id),
                'count': count,
                'percentage': percentage
            }
        
        return stats
    
    def compare_with_ground_truth(self) -> Optional[dict]:
        """
        Compare predictions with ground truth if available.
        
        :return: Dictionary with comparison results, or None if no ground truth
        """
        if self.ground_truth_data is None:
            logger.warning("No ground truth data available for comparison")
            return None
        
        prediction_stats = self.get_statistics()
        
        if self.ground_truth_detailed:
            gt_counts = defaultdict(int)
            if self.expected_crop_paths:
                for crop_path in self.expected_crop_paths:
                    entry = self._find_ground_truth(crop_path, os.path.basename(crop_path))
                    if entry:
                        gt_counts[entry['category_id']] += 1
            else:
                for entry in self.ground_truth_lookup.values():
                    gt_counts[entry['category_id']] += 1

            total_gt = sum(gt_counts.values())
            comparison = {
                'total_gt': total_gt,
                'total_pred': prediction_stats['total_detections'],
                'classes': {},
                'summary_metrics': {}
            }
            
            # Track errors for MAE and MAPE
            absolute_errors = []
            relative_errors = []
            matching_classes = 0
            
            class_ids = sorted(set(list(gt_counts.keys()) + list(prediction_stats['classes'].keys())))
            for class_id in class_ids:
                gt_count = gt_counts.get(class_id, 0)
                gt_percentage = (gt_count / total_gt * 100) if total_gt > 0 else 0.0
                
                pred_info = prediction_stats['classes'].get(class_id, {})
                pred_count = pred_info.get('count', 0)
                pred_percentage = pred_info.get('percentage', 0.0)
                
                class_name = self.category_manager.get_name(class_id)
                
                absolute_error = pred_count - gt_count
                relative_error = ((pred_count - gt_count) / gt_count * 100) if gt_count > 0 else 0.0
                
                comparison['classes'][class_id] = {
                    'name': class_name,
                    'gt_count': gt_count,
                    'pred_count': pred_count,
                    'gt_percentage': gt_percentage,
                    'pred_percentage': pred_percentage,
                    'absolute_error': absolute_error,
                    'relative_error': relative_error,
                    'found': pred_count > 0
                }
                
                if gt_count > 0:
                    absolute_errors.append(abs(absolute_error))
                    relative_errors.append(abs(relative_error))
                    if pred_count > 0:
                        matching_classes += 1
            
            if absolute_errors:
                comparison['summary_metrics'] = {
                    'mae': np.mean(absolute_errors),
                    'mape': np.mean(relative_errors) if relative_errors else 0.0,
                    'matching_classes': matching_classes,
                    'total_classes_in_gt': len(gt_counts),
                    'total_classes_in_pred': len(prediction_stats['classes']),
                    'total_count_error': comparison['total_pred'] - comparison['total_gt'],
                    'total_percentage_error': ((comparison['total_pred'] - comparison['total_gt']) / 
                                               comparison['total_gt'] * 100) if comparison['total_gt'] > 0 else 0.0
                }
            return comparison
        
        comparison = {
            'total_gt': sum(gt['count'] for gt in self.ground_truth_data.values()),
            'total_pred': prediction_stats['total_detections'],
            'classes': {},
            'summary_metrics': {}
        }
        
        # Track errors for MAE and MAPE
        absolute_errors = []
        relative_errors = []
        matching_classes = 0
        
        for class_id in sorted(set(list(self.ground_truth_data.keys()) + 
                                   list(prediction_stats['classes'].keys()))):
            
            gt_count = self.ground_truth_data.get(class_id, {}).get('count', 0)
            gt_percentage = self.ground_truth_data.get(class_id, {}).get('percentage', 0.0)
            
            pred_info = prediction_stats['classes'].get(class_id, {})
            pred_count = pred_info.get('count', 0)
            pred_percentage = pred_info.get('percentage', 0.0)
            
            class_name = self.category_manager.get_name(class_id)
            
            absolute_error = pred_count - gt_count
            relative_error = ((pred_count - gt_count) / gt_count * 100) if gt_count > 0 else 0.0
            
            comparison['classes'][class_id] = {
                'name': class_name,
                'gt_count': gt_count,
                'pred_count': pred_count,
                'gt_percentage': gt_percentage,
                'pred_percentage': pred_percentage,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'found': pred_count > 0
            }
            
            if gt_count > 0:
                absolute_errors.append(abs(absolute_error))
                relative_errors.append(abs(relative_error))
                if pred_count > 0:
                    matching_classes += 1
        
        if absolute_errors:
            comparison['summary_metrics'] = {
                'mae': np.mean(absolute_errors),
                'mape': np.mean(relative_errors) if relative_errors else 0.0,
                'matching_classes': matching_classes,
                'total_classes_in_gt': len(self.ground_truth_data),
                'total_classes_in_pred': len(prediction_stats['classes']),
                'total_count_error': comparison['total_pred'] - comparison['total_gt'],
                'total_percentage_error': ((comparison['total_pred'] - comparison['total_gt']) / 
                                           comparison['total_gt'] * 100) if comparison['total_gt'] > 0 else 0.0
            }
        
        return comparison
    
    def save_to_csv(self, output_path: Path):
        """
        Export results and ground truth comparison to CSV files.
        
        Creates:
        - {output_dir}/class_stats.csv: Overall class statistics
        - {output_dir}/predictions.csv: Individual predictions
        - {output_dir}/comparison.csv: Ground truth comparison
        
        :param output_path: Path where CSV will be saved
        """
        logger.info(f"Saving results to {output_path}")
        
        df_predictions = pd.DataFrame(self.predictions)
        if 'crop_path' in df_predictions.columns:
            df_predictions = df_predictions[['image_id', 'class_id', 'class_name', 'confidence']]
        
        # Create statistics dataframe
        stats = self.get_statistics()
        stats_data = []
        for class_id, class_stats in stats['classes'].items():
            stats_data.append({
                'class_id': class_id,
                'class_name': class_stats['name'],
                'count': class_stats['count'] if stats['total_detections'] > 0 else 0,
                'percentage': f"{class_stats['percentage']:.1f}%" if stats['total_detections'] > 0 else "0.0%"
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        # Write to CSV files
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(output_path.parent)
        
        df_stats.to_csv(output_path / "class_stats.csv", index=False)
        df_predictions.to_csv(output_path / "predictions.csv", index=False)
        logger.info(f"Results saved successfully")
        
        # Save comparison
        comparison = self.compare_with_ground_truth()
        if comparison:
            self._save_comparison_to_csv(comparison, output_path)
    
    def _save_comparison_to_csv(self, comparison: dict, output_path: Path):
        """
        Save detailed comparison results to CSV file.
        
        :param comparison: Dictionary from compare_with_ground_truth()
        :param output_path: Base output path
        """
        logger.info(f"Saving comparison results")
        
        # Prepare data for CSV
        comparison_data = []
        for class_id in sorted(comparison['classes'].keys()):
            cls_comp = comparison['classes'][class_id]
            comparison_data.append({
                'class_id': class_id,
                'class_name': cls_comp['name'],
                'gt_count': cls_comp['gt_count'],
                'pred_count': cls_comp['pred_count'],
                'gt_percentage': f"{cls_comp['gt_percentage']:.2f}%",
                'pred_percentage': f"{cls_comp['pred_percentage']:.2f}%",
                'absolute_error': cls_comp['absolute_error'],
                'relative_error': f"{cls_comp['relative_error']:.2f}%"
            })
        
        comparison_path = output_path / "comparison.csv"
        df = pd.DataFrame(comparison_data)
        df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison results saved successfully")
    
    def print_summary(self):
        """Print summary statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("FISH SPECIES IDENTIFICATION RESULTS SUMMARY")
        print("="*70)
        print(f"Total Detections: {stats['total_detections']}\n")
        
        if stats['total_detections'] == 0:
            print("No detections found.")
        else:
            print(f"{'Class ID':<12} {'Class Name':<30} {'Count':<10} {'Percentage':<10}")
            print("-"*70)
            
            for class_id in sorted(stats['classes'].keys()):
                class_info = stats['classes'][class_id]
                print(f"{class_id:<12} {class_info['name']:<30} "
                      f"{class_info['count']:<10} {class_info['percentage']:.2f}%")
        
        print("="*70 + "\n")
        
        # Print ground truth comparison if available
        comparison = self.compare_with_ground_truth()
        if comparison:
            self._print_comparison(comparison)
    
    def _print_comparison(self, comparison: dict):
        """
        Print comparison results to terminal.
        
        Args:
            comparison: Dictionary from compare_with_ground_truth()
        """
        print("\n" + "="*100)
        print("GROUND TRUTH COMPARISON RESULTS")
        print("="*100)
        
        print(f"\nTotal Counts:")
        print(f"  Ground Truth: {comparison['total_gt']} detections")
        print(f"  Predictions:  {comparison['total_pred']} detections")
        print(f"  Total Error:  {comparison['summary_metrics']['total_count_error']} "
              f"({comparison['summary_metrics']['total_percentage_error']:.2f}%)\n")
        
        print(f"Summary Metrics:")
        metrics = comparison['summary_metrics']
        print(f"  Mean Absolute Error (MAE):           {metrics['mae']:.2f} detections")
        print(f"  Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"  Classes Matched (GT ∩ Predictions): {metrics['matching_classes']}/{metrics['total_classes_in_gt']}")
        print(f"  Predicted Classes: {metrics['total_classes_in_pred']}\n")
        
        # Print per-class comparison
        print(f"{'Class':<12} {'Class Name':<25} {'GT Count':<10} {'Pred Count':<12} "
              f"{'Abs Error':<12} {'Rel Error':<12}")
        print("-"*100)
        
        for class_id in sorted(comparison['classes'].keys()):
            cls_comp = comparison['classes'][class_id]
            status = "✓" if cls_comp['found'] else "✗"
            
            print(f"{class_id:<12} {cls_comp['name']:<25} {cls_comp['gt_count']:<10} "
                  f"{cls_comp['pred_count']:<12} {cls_comp['absolute_error']:<12} "
                  f"{cls_comp['relative_error']:>10.2f}%")
        
        print("="*100 + "\n")

class InferencePipeline:
    """
    Main inference pipeline to process images and generate classification results.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the inference pipeline.
        
        :param config_path: Path to YAML configuration file
        """
        logger.info("Initializing Fish Species Identification Pipeline")
        
        self.config = InferenceConfig(config_path)
        self.category_manager = CategoryManager(self.config.categories_file, logger)
        self.image_processor = None

        self.model = ResNetInferenceModel(
            self.config.weights_path,
            num_classes=len(self.category_manager.categories),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.output = ResultsManager(
            self.category_manager,
            ground_truth_csv=self.config.ground_truth_csv
        )
        
        logger.info("Pipeline initialized successfully")

    def _normalize_crop_relative_path(self, crop_path: Path) -> str:
        """Normalize the crop path relative to the configured crop root."""
        relative_path = crop_path.relative_to(self.config.crops_dir)
        normalized = os.path.normpath(str(relative_path)).replace('\\', '/')
        split_name = self.config.crops_dir.name.lower()
        if split_name in ('train', 'val', 'test'):
            normalized = os.path.normpath(os.path.join(split_name, normalized)).replace('\\', '/')
        return normalized
    
    def run(self):

        logger.info("="*70)
        logger.info("STARTING INFERENCE PIPELINE")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info("="*70)
        
        processed_count = 0
        error_count = 0

        # Walk through crop directory and process each image recursively
        crop_files = []
        for root, _, files in os.walk(self.config.crops_dir):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    crop_files.append(Path(root) / file)

        logger.info(f"Found {len(crop_files)} crop files")

        # Register all crop paths that should be compared to ground truth
        for crop_path in crop_files:
            relative_crop_path = self._normalize_crop_relative_path(crop_path)
            self.output.register_crop_path(relative_crop_path)

        # Process crops in batches
        batch_crops = []
        batch_image_ids = []
        batch_crop_paths = []
        
        for crop_path in tqdm(crop_files, desc="Processing cropped images"):
            try:
                image_id = crop_path.stem
                image = cv2.imread(str(crop_path))
                if image is None:
                    logger.warning(f"Failed to load crop image: {crop_path}")
                    continue
                
                batch_crops.append(image)
                batch_image_ids.append(image_id)
                batch_crop_paths.append(self._normalize_crop_relative_path(crop_path))
                
                # Process batch when it reaches batch_size
                if len(batch_crops) >= self.config.batch_size:
                    self._process_crop_batch(batch_crops, batch_image_ids, batch_crop_paths)
                    processed_count += len(batch_crops)
                    batch_crops = []
                    batch_image_ids = []
                    batch_crop_paths = []
                    
            except Exception as e:
                logger.warning(f"Error processing crop {crop_path.name}: {e}")
                error_count += 1
        
        # Process remaining crops in final batch
        if batch_crops:
            self._process_crop_batch(batch_crops, batch_image_ids, batch_crop_paths)
            processed_count += len(batch_crops)
        
        logger.info("="*70)
        logger.info("INFERENCE COMPLETE")
        logger.info(f"Successfully processed: {processed_count} items")
        logger.info(f"Errors encountered: {error_count}")
        logger.info("="*70)
        
        self.output.save_to_csv(self.config.output_dir)
        self.output.print_summary()
    
    def _process_crop_batch(self, crops: list[np.ndarray], image_ids: list[str], crop_paths: list[str]):
        """
        Process a batch of cropped images.

        :param crops: List of crop images as numpy arrays
        :param image_ids: List of corresponding image IDs
        :param crop_paths: List of corresponding relative crop paths for ground truth comparison
        """
        predictions = self.model.predict(crops)
        
        for image_id, crop_path, (predicted_class, confidence) in zip(image_ids, crop_paths, predictions):
            self.output.add_prediction(image_id, predicted_class, confidence, crop_path=crop_path)
    
    def _process_crop(self, crop_path: Path):
        """
        Process a single cropped image directly.

        :param crop_path: Path to a pre-extracted crop image file
        """
        
        image_id = crop_path.stem
        image = cv2.imread(str(crop_path))
        if image is None:
            raise ValueError(f"Failed to load crop image: {crop_path}")

        predicted_class, confidence = self.model.predict(image)
        self.output.add_prediction(
            image_id,
            predicted_class,
            confidence,
            crop_path=self._normalize_crop_relative_path(crop_path)
        )

def main():

    parser = argparse.ArgumentParser(description='Fish Species Identification Inference Pipeline')
    
    parser.add_argument('--config', type=str, default='config/fish_identification_cfg.yml', help='Path to YAML configuration file (default: config/inference_config.yml)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (DEBUG level)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    try:
        pipeline = InferencePipeline(args.config)
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        raise


if __name__ == '__main__':
    main()
