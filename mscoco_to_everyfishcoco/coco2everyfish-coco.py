#!/usr/bin/env python3
"""
Convert MS-COCO format annotations to EVERYFISH-COCO format.

This script takes a directory containing a JSON file with MS-COCO annotations and a folder
containing images, then restructures them into:
- annotations/ folder with one JSON file per image containing image metadata and annotations
- images/ folder containing all images (copied from source)
- info/ folder containing metadata files (categories.json, licenses.json, info.json)

Usage:
    python3 ms-coco2everyfish-coco.py --coco_json <coco_json_path> --images_folder <images_folder> --output_folder <output_folder>
"""

import argparse
import json
import logging
import shutil
import re
import pandas as pd
import yaml

from pathlib import Path
from collections import defaultdict
from typing import Any, Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging import setup_logging

logger = setup_logging("INFO")

class MSCOCOToEveryFishCOCOConverter:
    """
    Converts MS-COCO format annotations to EVERYFISH-COCO format.
    
    This converter:
    1. Reads a single MS-COCO JSON file
    2. Walks through the images folder to find all images (including in subfolders)
    3. Creates output folder structure with annotations, images, and info directories
    4. Generates one JSON file per image with image metadata and associated annotations
    """
    
    def __init__(self, coco_json_path: str, images_folder: str, output_folder: str):
        """
        :param coco_json_path: Path to the MS-COCO format JSON file
        :param images_folder: Path to the folder containing images (can have subfolders)
        :param output_folder: Path where the new structure will be created
        """
        
        self.coco_json_path = Path(coco_json_path)
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.config_file = Path(__file__).resolve().parent.parent / 'config' / "info_metadata.yml"
        
        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON file not found: {self.coco_json_path}")
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        
        # Load MS-COCO data
        self.coco_data = self._load_coco_json()
        
        # Load YAML config file
        self.config = self._load_yaml_config()
        
        # Load FAO codes mapping from Excel file
        self.fao_code_mapping, self.scientific_fao_mapping = self._load_fao_codes()
        
        # Create internal mappings for faster lookup
        self.images_by_id = {img['id']: img for img in self.coco_data.get('images', [])}
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            self.annotations_by_image[ann['image_id']].append(ann)
        
        self.categories_by_id = {cat['id']: cat for cat in self.coco_data.get('categories', [])}
        self.licenses_by_id = {lic['id']: lic for lic in self.coco_data.get('licenses', [])}
        
        logger.info(f"Loaded MS-COCO JSON with {len(self.images_by_id)} images")
        logger.info(f"Total annotations: {len(self.coco_data.get('annotations', []))}")
        logger.info(f"Total categories: {len(self.categories_by_id)}")
    
    def _load_coco_json(self) -> dict[str, Any]:
        """        
        :return: dictionary containing MS-COCO data 
        """
        
        try:
            with open(self.coco_json_path, 'r') as f:
                data = json.load(f)
            
            required_keys = ['images', 'annotations', 'categories']
            missing = [key for key in required_keys if key not in data]
            if missing:
                logger.warning(f"MS-COCO JSON missing keys: {missing}. These will be treated as empty.")
            
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.coco_json_path}: {e}")
    
    def _load_fao_codes(self) -> tuple[dict[str, str], dict[str, str]]:
        """
        Load FAO codes mapping from Excel file
        Creates mappings from english or scientific name to FAO Alpha3 code
       
        :return: Tuple of (english_mapping, scientific_mapping) where each is a dict mapping name to Alpha3_Code            
        """
        
        logger.info("Loading FAO codes from Excel file...")

        fao_path = Path(__file__).resolve().parent.parent / 'assets' / 'FAO-CODES.xlsx'
        
        if fao_path.exists():
            fao_file = fao_path
        else:
            logger.warning(f"FAO-CODES.xlsx not found in any search location. FAO codes will not be added.")
            return {}, {}
        
        try:
            df = pd.read_excel(fao_file)
            
            english_column = 'English_name'            
            scientific_column = 'Scientific_Name'
            
            if (scientific_column is None and english_column is None) or 'Alpha3_Code' not in df.columns:
                raise ValueError(f"Excel file must contain 'English_name'/'Scientific_Name' and 'Alpha3_Code' columns")
            
            # Create mapping: English name (lowercase) -> Alpha3_Code
            # Skip empty values
            english_names = df[english_column].astype(str).str.strip().str.lower()
            alpha_codes = df['Alpha3_Code'].astype(str).str.strip()
            
            mapping = {}
            scientific_mapping = {}
            
            for name, code in zip(english_names, alpha_codes):
                if pd.isnull(name):
                    continue
                if name not in mapping:
                    mapping[name] = code
                else:
                    logger.debug(f"Duplicate English_name '{name}' found. Using first occurrence: '{mapping[name]}', ignoring: '{code}'")
            
            # Create Scientific name mapping if available
            if scientific_column:
                scientific_names = df[scientific_column].astype(str).str.strip().str.lower()
                for scientific_name, code in zip(scientific_names, alpha_codes):
                    if pd.isnull(scientific_name):
                        continue
                    if scientific_name not in scientific_mapping:
                        scientific_mapping[scientific_name] = code
                    else:
                        logger.debug(f"Duplicate Scientific_name '{scientific_name}' found. Using first occurrence: '{scientific_mapping[scientific_name]}', ignoring: '{code}'")
                
                logger.info(f"Loaded {len(scientific_mapping)} Scientific name mappings")
            
            logger.info(f"Loaded {len(mapping)} unique FAO code mappings from {fao_file} (duplicates removed)")
            return mapping, scientific_mapping
        except Exception as e:
            logger.warning(f"Error loading FAO codes from Excel: {e}. FAO codes will not be added.")
            return {}, {}, {}
    
    def _load_yaml_config(self) -> dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        The YAML file should contain:
        - 'info': dict with dataset metadata (for info.json)
        - 'licenses': list of dicts with license information (for licenses.json)
        - 'metadata': dict with template fields for metadata.json (e.g., gear_type, mesh_selectivity_device, etc.)
        
        :return: dictionary containing the loaded config, or empty dict if file doesn't exist            
        """

        if not self.config_file or not self.config_file.exists():
            logger.warning(f"YAML config file not found: {self.config_file}. Using defaults.")
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded YAML config from {self.config_file}")
            return config or {}
        except Exception as e:
            logger.warning(f"Error loading YAML config from {self.config_file}: {e}. Using defaults.")
            return {}
    
    def _select_fao_code(self, category_name: str, candidates: list[tuple[str, str]]) -> Optional[str]:
        """
        Ask the user to select among multiple FAO candidates.
        This can occur if fish names are ambiguous and match multiple entries in the FAO code mappings.
        """
        
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]

        prompt_lines = [
            f"Multiple FAO matches found for '{category_name}':"
        ]
        for idx, (english_name, code) in enumerate(candidates, start=1):
            prompt_lines.append(f"  {idx}) {english_name} -> {code}")
        prompt_lines.append("Enter the number of the desired match: ")

        while True:
            choice = input("\n".join(prompt_lines)).strip()
            if choice.isdigit():
                selected = int(choice)
                if 1 <= selected <= len(candidates):
                    return candidates[selected - 1][1]
            print(f"Invalid selection '{choice}'. Please enter a number between 1 and {len(candidates)}.")

    def _find_images(self) -> dict[str, Path]:
        """
        Walk through the images folder to find all image files.
        Supports: jpg, jpeg, png
        
        :return: dictionary mapping filename or folder/filename to full path
        """
        image_extensions = {'.jpg', '.jpeg', '.png'}
        images = {}
        
        for image_path in self.images_folder.rglob('*'):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                relative_path = image_path.relative_to(self.images_folder)
                images[str(relative_path)] = image_path  # key by relative path like "folder/image.png"
                if image_path.name not in images:
                    images[image_path.name] = image_path  # key by filename only, if not already present
        
        logger.info(f"Found {len(images)} images in {self.images_folder}")
        return images
    
    def _create_per_image_annotation(self, image_info: dict[str, Any], annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Create the EVERYFISH-COCO format from MS-COCO image and annotation data.
        
        :param image_info: Image metadata from MS-COCO file
        :param annotations: list of annotations for this image from MS-COCO file
        :return: dictionary with 'image' and 'annotation' sections in the new format
        """
        # 'image' section template
        image_section = {
            'license': image_info.get('license', None), # int (This is an ID from the “licenses.json”)
            'file_name': image_info.get('file_name', ''),
            'width': image_info.get('width', 0),
            'height': image_info.get('height', 0),
            'date_captured': image_info.get('date_captured', ''),
            'image_id': image_info.get('id', 0),
            'annotations_type': self._extract_annotations_types(annotations) # [detection, segmentation, weight_estimation, length_estimation]
        }
        
        # 'annotation' section template
        annotation_list = []
        for idx, ann in enumerate(annotations):
            annotation_entry = {
                'id': idx,  # Sequential ID for objects in image
                'image_id': ann.get('image_id', image_info.get('id', 0)),
                'category_id': int(ann.get('category_id', None)),
                'bbox': ann.get('bbox', None),  #  [x, y, width, height] (x and y coordinates from the top left) 
                'segmentation': ann.get('segmentation', None),  # A list of list of points
                'area': ann.get('area', None), # float (area of the polygon in pixels)
                'fish_length': ann.get('length', None),  # float (in cm)
                'fish_weight': ann.get('weight', None),  # float (in kg)
                'occlusion_type': ann.get('occlusion_type', None),  # int [0(fully visible),  1(partially visible), 2(immeasurable) ]
                'annotator_type': ann.get('annotator_type', "unknown")  #  [‘human’, ‘machine’ or ‘unknown’]
            }
            annotation_list.append(annotation_entry)
        
        return {
            'image': image_section,
            'annotation': annotation_list
        }
    
    def _extract_annotations_types(self, annotations: list[dict[str, Any]]) -> list[str]:
        """
        Extract annotation types from annotations list.
        Determines which annotation types are present based on which fields contain data.
        
        :param annotations: list of annotations for an image
        :return: list of annotation type strings
        """
        
        types = set()
        
        for ann in annotations:
            if ann.get('segmentation'):
                types.add('segmentation')
            if ann.get('bbox'):
                types.add('detection')
            if ann.get('weight'):
                types.add('weight_estimation')
            if ann.get('length'):
                types.add('length_estimation')
        
        return sorted(list(types))
    
    def _setup_output_structure(self) -> tuple:
        """
        Create the output folder structure.
        
        Creates three main directories:
        - annotations/: Will contain JSON files (one per image)
        - images/: Will contain image files
        - info/: Will contain metadata files (categories.json, licenses.json, info.json, metadata.json)
        
        :return: Tuple of (annotations_dir, images_dir, info_dir) Path objects
        """

        if self.output_folder.exists(): # Ask user to confirm overwrite if output folder already exists
            logger.warning(f"Output folder already exists: {self.output_folder}")
            response = input("Do you want to overwrite it? (y/[n]): ").strip().lower()
            if response == 'y':
                shutil.rmtree(self.output_folder)
                logger.info(f"Removed existing folder: {self.output_folder}")
            else:
                raise FileExistsError(f"Output folder already exists: {self.output_folder}")
        
        annotations_dir = self.output_folder / 'annotations'
        images_dir = self.output_folder / 'images'
        info_dir = self.output_folder / 'info'
        
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output structure in {self.output_folder}")
        return annotations_dir, images_dir, info_dir
    
    def create_info_template():
        # info.json
        INFO_TEMPLATE = {'year': None,
                        'version': None,
                        'description': None,
                        'contributor': None,
                        'url': None,
                        'date_created': None,
                        'sharing_availability': None}

        # licenses.json
        LICENSES_TEMPLATE = {'id': None,
                            'name': None,
                            'url': None}

        # metadata.json
        METADATA_TEMPLATE = {'image_name': None,
                            'gear_type': None,
                            'mesh_selectivity_device': None,
                            'targeted_fish': None,
                            'fish_region': None}
        
        return INFO_TEMPLATE, LICENSES_TEMPLATE, METADATA_TEMPLATE
    
    def _save_info_files(self, info_dir: Path) -> None:
        """
        Save metadata files (except categories.json) to the info folder from YAML config or defaults.
        
        Creates:
        - info.json: Dataset information from config or defaults
        - licenses.json: License information from config or defaults
        - metadata.json: Per-image metadata list with image_name and config template fields
        
        :param info_dir: Path to the info directory
        """
        
        # Load info data from config, fall back to template
        if self.config and 'info' in self.config:
            info_data = self.config['info']
            logger.info("Loaded info from YAML config")
        else:
            info_data = self.coco_data.get('info', {})
            logger.info("Loaded info from template")
        
        with open(info_dir / 'info.json', 'w') as f:
            json.dump(info_data, f, indent=2)
        logger.info("Saved info.json")
        
        # Load licenses data from config, fall back to template
        if self.config and 'licenses' in self.config:
            licenses_data = self.config['licenses']
            logger.info("Loaded licenses from YAML config")
        else:
            licenses_data = self.coco_data.get('licenses', [])
            logger.info("Loaded licenses from template")

        licenses_list = []
        for lic in licenses_data:
            license_entry = {
                'id': lic.get('id', None),
                'name': lic.get('name', None),
                'url': lic.get('url', None)
            }
            licenses_list.append(license_entry)
        
        with open(info_dir / 'licenses.json', 'w') as f:
            json.dump(licenses_list, f, indent=2)
        logger.info(f"Saved licenses.json with {len(licenses_list)} licenses")
        
        # Create metadata.json with per-image entries
        metadata_template = {}
        if self.config and 'metadata' in self.config:
            metadata_template = self.config['metadata']
            logger.info("Loaded metadata template from YAML config")
        else:
            # Use default template
            metadata_template = {
                'gear_type': None,
                'mesh_selectivity_device': None,
                'targeted_fish': None,
                'fish_region': None
            }
            logger.info("Using default metadata template")
        
        # Create one metadata entry per image with image_name and template fields
        metadata_list = []
        for image_info in self.coco_data.get('images', []):
            metadata_entry = dict(metadata_template)  # Copy template
            metadata_entry['image_name'] = image_info.get('file_name', '')
            metadata_list.append(metadata_entry)
        
        with open(info_dir / 'metadata.json', 'w') as f:
            json.dump(metadata_list, f, indent=2)
        logger.info(f"Saved metadata.json with {len(metadata_list)} image entries")
    
    def _save_category_file(self, info_dir: Path) -> None:
        """
        Save category file to the info folder.
        
        Creates:
        - categories.json: Category information from MS-COCO file with FAO codes
        
        :param info_dir: Path to the info directory
        """

        # Save categories with FAO codes
        categories_data = self.coco_data.get('categories', [])

        allowed_keys = {'id', 'name', 'FAO'}
        
        for i, category in enumerate(categories_data):
            category = {k: v for k, v in category.items() if k in allowed_keys} # delete unwanted keys

            category_name = category.get('name', '').lower().strip()
            category_name_normalized = category_name.replace('_', ' ')
            
            # Check if FAO code already exists in MS-COCO annotation
            if 'FAO' in category.keys():
                logger.debug(f"Category '{category.get('name')}' already has FAO code: '{category['FAO']}'")
                continue
            
            # Otherwise first try exact English match, then English word match, then Scientific matches
            fao_code = None
            
            # Try exact English match
            for english_name, code in self.fao_code_mapping.items():
                if not english_name:
                    continue
                if category_name_normalized == english_name:
                    fao_code = code
                    logger.debug(f"Exact English match for category '{category.get('name')}' -> '{english_name}' -> '{code}'")
                    break
            
            # Try English word match
            if fao_code is None:
                word_pattern = re.compile(rf"\b{re.escape(category_name_normalized)}\b")
                candidates = []
                for english_name, code in self.fao_code_mapping.items():
                    if not english_name:
                        continue
                    if word_pattern.search(english_name):
                        candidates.append((english_name, code))
                
                if len(candidates) == 1:
                    fao_code = candidates[0][1]
                    logger.debug(f"Single English word match for category '{category.get('name')}' -> '{candidates[0][0]}' -> '{fao_code}'")
                elif len(candidates) > 1:
                    fao_code = self._select_fao_code(category.get('name', ''), candidates)
                    logger.debug(f"Selected English FAO code '{fao_code}' for category '{category.get('name')}' from candidates: {candidates}")
            
            # Try Scientific name matches if no English match found
            if fao_code is None and self.scientific_fao_mapping:
                # Try exact Scientific match
                for scientific_name, code in self.scientific_fao_mapping.items():
                    if not scientific_name:
                        continue
                    if category_name_normalized == scientific_name:
                        fao_code = code
                        logger.debug(f"Exact Scientific match for category '{category.get('name')}' -> '{scientific_name}' -> '{code}'")
                        break
                
                # Try Scientific word match
                if fao_code is None:
                    word_pattern = re.compile(rf"\b{re.escape(category_name_normalized)}\b")
                    candidates = []
                    for scientific_name, code in self.scientific_fao_mapping.items():
                        if not scientific_name:
                            continue
                        if word_pattern.search(scientific_name):
                            candidates.append((scientific_name, code))
                    
                    if len(candidates) == 1:
                        fao_code = candidates[0][1]
                        logger.debug(f"Single Scientific word match for category '{category.get('name')}' -> '{candidates[0][0]}' -> '{fao_code}'")
                    elif len(candidates) > 1:
                        fao_code = self._select_fao_code(category.get('name', ''), candidates)
                        logger.debug(f"Selected Scientific FAO code '{fao_code}' for category '{category.get('name')}' from candidates: {candidates}")
                
                # Try abbreviated Scientific name expansion (e.g., "S. maximus" -> "Scophthalmus maximus")
                if fao_code is None:
                    abbrev_match = re.match(r'^([a-z])\.\s+(.+)$', category_name_normalized)
                    if abbrev_match:
                        genus_initial, species = abbrev_match.groups()
                        
                        # Look for abbreviated Scientific names
                        candidates = []
                        for scientific_name, code in self.scientific_fao_mapping.items():
                            if not scientific_name:
                                continue
                            if scientific_name.startswith(genus_initial) and species in scientific_name:
                                candidates.append((scientific_name, code))
                        
                        if len(candidates) == 1:
                            fao_code = candidates[0][1]
                            logger.debug(f"Abbreviated Scientific match for category '{category.get('name')}' -> '{candidates[0][0]}' -> '{fao_code}'")
                        elif len(candidates) > 1:
                            fao_code = self._select_fao_code(category.get('name', ''), candidates)
                            logger.debug(f"Selected abbreviated Scientific FAO code '{fao_code}' for category '{category.get('name')}' from candidates: {candidates}")

            # Set FAO code in category entry, or empty string if not found
            if fao_code:
                category['FAO'] = fao_code
            else:
                category['FAO'] = ""
            
            categories_data[i] = category  # Update the category entry with FAO code
            logger.debug(f"FAO code for category '{category.get('name')}': '{category['FAO']}'")

        # Ensure categories are output in ascending order by id
        categories_data = sorted(categories_data, key=lambda cat: cat.get('id', 0))

        with open(info_dir / 'categories.json', 'w') as f:
            json.dump(categories_data, f, indent=2)
        logger.info(f"Saved categories.json with {len(categories_data)} image entries")
    
    def convert(self) -> None:
        """
        This method:
        1. Sets up the output folder structure
        2. Finds all images in the source folder
        3. For each image in MS-COCO file, creates corresponding annotation JSON
        4. Copies images to the output images folder
        5. Saves metadata files        
        """
        
        logger.info("=" * 80)
        logger.info("Starting MS-COCO to EVERYFISH-COCO conversion")
        logger.info("=" * 80)
        
        annotations_dir, images_dir, info_dir = self._setup_output_structure()
        
        found_images = self._find_images()
        
        # Process each image from MS-COCO file
        processed = 0
        skipped = 0
        
        for image_id, image_info in tqdm(self.images_by_id.items(), unit='images', leave=True, desc='Processing images'):
            filename = image_info.get('file_name', '')
            
            if not filename:
                logger.warning(f"Skipping image {image_id}: no file_name")
                skipped += 1
                continue
            
            # Get image annotations
            annotations = self.annotations_by_image.get(image_id, [])
            
            # Create per-image annotation
            per_image_annotation = self._create_per_image_annotation(image_info, annotations)
            
            # Save annotation JSON
            annotation_filename = f"{Path(filename.replace('/', '_')).stem}.json"
            annotation_path = annotations_dir / annotation_filename
            with open(annotation_path, 'w') as f:
                json.dump(per_image_annotation, f, indent=2)
            
            # Copy image if found
            if filename in found_images:
                source_image = found_images[filename]
                dest_image = images_dir / filename
                
                dest_image.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(source_image, dest_image)
                logger.debug(f"Copied image: {filename}")
            else:
                logger.warning(f"Image not found in source folder: {filename}")
                    
        # Save metadata files
        self._save_category_file(info_dir)
        self._save_info_files(info_dir)
        
        logger.info("=" * 80)
        logger.info(f"Conversion completed successfully!")
        logger.info(f"  Total images processed: {processed}")
        logger.info(f"  Total images skipped: {skipped}")
        logger.info(f"  Output directory: {self.output_folder}")
        logger.info("=" * 80)


def main():

    parser = argparse.ArgumentParser(description='Convert MS-COCO format annotations to EVERYFISH-COCO format')
    
    parser.add_argument('--coco_json', type=str, required=True, help='Path to the MS-COCO format JSON file')
    parser.add_argument('--images', type=str, required=True, help='Path to the folder containing images (can have subfolders)')
    parser.add_argument('--output', type=str, required=True, help='Path where the new structure will be created')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    try:
        converter = MSCOCOToEveryFishCOCOConverter(args.coco_json, args.images, args.output)
        converter.convert()
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
