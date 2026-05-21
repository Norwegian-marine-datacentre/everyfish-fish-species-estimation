
import logging
from pathlib import Path
import json
from typing import Optional
from collections import defaultdict
import cv2
import numpy as np

class CategoryManager:
    """Manages fish species categories and their IDs."""

    def __init__(self, categories_file: Path, logger: logging.Logger):
        """Load categories from JSON file."""
        if not categories_file.exists():
            raise FileNotFoundError(f"Categories file not found: {categories_file}")

        logger.info(f"Loading categories from {categories_file}")

        with open(categories_file, 'r') as f:
            data = json.load(f)

        # Build ID -> name mapping
        self.categories = {}
        if isinstance(data, dict) and 'categories' in data:
            for cat in data['categories']:
                self.categories[cat['id']] = cat['name']
        elif isinstance(data, list):
            for cat in data:
                self.categories[cat['id']] = cat['name']
        else:
            raise ValueError("Unexpected categories file format")

        logger.info(f"Loaded {len(self.categories)} categories")
        for cat_id, cat_name in sorted(self.categories.items()):
            logger.debug(f"  Category {cat_id}: {cat_name}")

    def get_name(self, category_id: int) -> str:
        """Get category name by ID."""
        return self.categories.get(category_id, f"Unknown_{category_id}")

    def get_id(self, category_name: str) -> Optional[int]:
        """Get category ID by name."""
        for cat_id, cat_name in self.categories.items():
            if cat_name == category_name:
                return cat_id
        return None
    

class AnnotationLoader:
    """Loads and manages EVERYFISH-COCO format annotations."""

    def __init__(self, annotation_file: Path, logger: logging.Logger):
        """Load COCO format annotations from JSON file."""
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        logger.info(f"Loading annotations from {annotation_file}")

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_metadata = self.coco_data.get('image', [])
        self.annotations_by_image_id = defaultdict(list)

        for ann in self.coco_data.get('annotation', []):
            self.annotations_by_image_id[ann['image_id']].append(ann)

        logger.info(f"Loaded 1 image with "
                    f"{len(self.coco_data.get('annotation', []))} total annotations")

    def get_image_annotations(self, image_id: int) -> list[dict]:
        """Get all annotations for a specific image."""
        return self.annotations_by_image_id.get(image_id, [])

    def get_image_metadata(self) -> list[dict]:
        """Get all image metadata."""
        return self.image_metadata
    
class ImageProcessor:
    """Handles image loading and annotation-based cropping."""

    def __init__(self, images_dir: Path):
        """Initialize image processor."""
        self.images_dir = images_dir

    def load_image(self, image_filename: str) -> np.ndarray:
        """Load image from file."""

        image_path = self.images_dir / image_filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image  # BGR format

    def crop_by_bbox(self, image: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Crop image using bounding box (detection format)."""
        x, y, w, h = [int(v) for v in bbox]
        x2 = x + w
        y2 = y + h

        # Ensure coordinates are within image bounds
        h_img, w_img = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        crop = image[y:y2, x:x2]
        return crop

    def crop_by_segmentation(self, image: np.ndarray,
                            segmentation: list[float],
                            make_transparent: bool = True) -> np.ndarray:
        """
        Crop image using segmentation mask with transparent background.

        Creates an RGBA image where the fish area is visible and the background
        (area outside the polygon mask) is transparent.
        """

        # Convert flat list to coordinate pairs
        coords = np.array(segmentation).reshape(-1, 2).astype(np.int32)

        # Get bounding box from polygon
        x_min = max(0, coords[:, 0].min())
        y_min = max(0, coords[:, 1].min())
        x_max = min(image.shape[1], coords[:, 0].max())
        y_max = min(image.shape[0], coords[:, 1].max())

        # Crop the region
        crop = image[y_min:y_max, x_min:x_max]

        # Adjust coordinates relative to cropped region
        adjusted_coords = coords - np.array([x_min, y_min])

        # Create binary mask from polygon
        mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [adjusted_coords], 255)

        # Create RGBA image (add alpha channel)
        if make_transparent:
            if crop.shape[2] == 3:  # BGR image
                rgba_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            else:
                rgba_crop = crop.copy()

            # Apply mask to alpha channel (make background transparent)
            rgba_crop[:, :, 3] = mask
        else:
            rgba_crop = crop.copy()

        return rgba_crop