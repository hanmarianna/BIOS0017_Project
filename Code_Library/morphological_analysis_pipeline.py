#!/usr/bin/env python3
"""
Morphological Segmentation Analysis Pipeline (GT vs CP vs Watershed) comparing Areas and Perimeters.

This pipeline processes a batch of images and computes per-cell area and perimeter metrics for comparing
ground truth (GT), Cellpose Checkpoint (CP), and watershed (WS) segmentation outputs.
"""

import os
import glob
import argparse
import logging
import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.transform import resize

# Suppress deprecation warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Configure logging
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove existing handlers
while logger.handlers:
    logger.handlers.pop()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler("debug.log", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Helper Functions
def recursive_extract(data):
    """Recursively extract the first NumPy array from nested structures."""
    if isinstance(data, np.ndarray) and data.dtype != np.dtype('O'):
        return data
    if isinstance(data, dict):
        for value in data.values():
            result = recursive_extract(value)
            if result is not None:
                return result
    if isinstance(data, list):
        for item in data:
            result = recursive_extract(item)
            if result is not None:
                return result
    return None

def load_image(filepath):
    """Load an image and convert to grayscale if needed."""
    try:
        image = imread(filepath)
        if image.ndim == 3:
            image = image[..., 0]
        return image
    except Exception as e:
        logging.error(f"Error loading image from {filepath}: {e}")
        raise

def load_binary_mask(filepath, threshold=0):
    """Load and binarize a mask image."""
    try:
        image = imread(filepath)
    except ValueError as e:
        if 'not supported' in str(e):
            from PIL import Image
            image = np.array(Image.open(filepath).convert('L'))
        else:
            raise
    binary = image > threshold
    logging.info(f"Binary mask stats for {filepath}: shape={binary.shape}, sum={np.sum(binary)}")
    return binary

def load_instance_mask(filepath):
    """Load an instance-based mask from a .npy file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        
        if isinstance(data, dict):
            inst_mask = data.get("masks", recursive_extract(data))
        else:
            inst_mask = recursive_extract(data)
        
        if inst_mask is None or not isinstance(inst_mask, np.ndarray):
            raise ValueError("Could not extract valid instance mask")
        return inst_mask.astype(np.int32)
    except Exception as e:
        logging.error(f"Error loading instance mask from {filepath}: {e}")
        raise

def ensure_same_shape(reference, mask):
    """Ensure mask has same shape as reference image."""
    if reference.shape != mask.shape:
        logging.warning(f"Shape mismatch: reference {reference.shape} vs mask {mask.shape}")
        mask_resized = resize(mask, reference.shape, order=0, preserve_range=True, anti_aliasing=False)
        return (mask_resized > 0.5)
    return mask

# Metric Calculation Functions
def compute_basic_metrics(mask):
    """
    Compute basic shape metrics for a binary mask.
    Returns only area and perimeter
    """
    lab = label(mask.astype(np.uint8))
    props = regionprops(lab)
    if not props:
        return {}
    cell = props[0]
    return {"area": cell.area, "perimeter": cell.perimeter}

def get_accurate_cell_count(instance_mask, method, base):
    """
    Get accurate cell count from instance segmentation mask.
    """
    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (0)
    count = len(unique_labels)
    
    logging.info(f"{base} - {method} analysis:")
    logging.info(f"  Total cells: {count}")
    logging.info(f"  Label range: {unique_labels.min()} to {unique_labels.max()}")
    
    return count, unique_labels

def detailed_cell_analysis_three_way(full_image, gt_inst, gt_bin, cp_inst, cp_bin, ws_mask, 
                                   patch_size=100, base="unknown", debug_mode=False, output_dir="output"):
    """
    Analyze cells using GT centroids as reference points, comparing against both CP and WS.
    Only area and perimeter metrics are extracted.
    """
    total_cells, unique_labels = get_accurate_cell_count(gt_inst, "GT", base)
    
    results = []
    half_patch = patch_size // 2

    for label_id in unique_labels:
        cell_mask = gt_inst == label_id
        coords = np.where(cell_mask)
        if len(coords[0]) == 0:
            continue
        
        r = int(np.mean(coords[0]))
        c = int(np.mean(coords[1]))
        
        r0 = max(r - half_patch, 0)
        r1 = min(r + half_patch, full_image.shape[0])
        c0 = max(c - half_patch, 0)
        c1 = min(c + half_patch, full_image.shape[1])

        if (r1 - r0) < patch_size or (c1 - c0) < patch_size:
            logging.info(f"Skipping cell {label_id} near edge")
            continue

        patch_gt = gt_bin[r0:r1, c0:c1]
        patch_cp = cp_bin[r0:r1, c0:c1]
        patch_ws = ws_mask[r0:r1, c0:c1]
        
        # Skip if GT patch is empty
        if np.sum(patch_gt) == 0:
            logging.warning(f"GT binary patch is empty for cell {label_id}")
            continue

        # Extract metrics and store results
        metrics_gt = compute_basic_metrics(patch_gt)
        metrics_cp = compute_basic_metrics(patch_cp)
        metrics_ws = compute_basic_metrics(patch_ws)
        
        # Collect only area and perimeter metrics
        result = {
            "Cell_ID": label_id,
            "Centroid_row": r,
            "Centroid_col": c,
            "GT_area": metrics_gt.get("area", np.nan),
            "CP_area": metrics_cp.get("area", np.nan), 
            "WS_area": metrics_ws.get("area", np.nan),
            "GT_perimeter": metrics_gt.get("perimeter", np.nan),
            "CP_perimeter": metrics_cp.get("perimeter", np.nan),
            "WS_perimeter": metrics_ws.get("perimeter", np.nan),
        }
        
        results.append(result)

    # Log summary statistics
    processed_cells = len(results)
    logging.info(f"Image {base} summary:")
    logging.info(f"Processed cells: {processed_cells}")

    return pd.DataFrame(results), processed_cells, total_cells

def main():
    parser = argparse.ArgumentParser(
        description="Final Segmentation Analysis Pipeline (GT vs CP vs WS) - Area and Perimeter Only"
    )
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw fluorescence images (TIFF).")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth masks.")
    parser.add_argument("--cp_dir", required=True, help="Directory containing Cellpose masks.")
    parser.add_argument("--ws_dir", required=True, help="Directory containing watershed masks.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    parser.add_argument("--patch_size", type=int, default=100, help="Size of patches for analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.tif")))
    overall_results = []
    detailed_results_list = []

    for raw_path in raw_paths:
        base = os.path.splitext(os.path.basename(raw_path))[0]
        logging.info(f"Processing image: {base}")

        # Load all required images
        try:
            intensity_img = load_image(raw_path)
            gt_inst = load_instance_mask(os.path.join(args.gt_dir, f"{base}_seg.npy"))
            gt_bin = load_binary_mask(os.path.join(args.gt_dir, f"{base}_cp_masks_binary.png"))
            cp_inst = load_instance_mask(os.path.join(args.cp_dir, f"{base}_seg.npy"))
            cp_bin = load_binary_mask(os.path.join(args.cp_dir, f"{base}_cp_masks_binary.png"))
            ws_mask = load_binary_mask(os.path.join(args.ws_dir, f"{base}_watershed_matlab.tif"))

            # Ensure all masks match the intensity image shape
            gt_bin = ensure_same_shape(intensity_img, gt_bin)
            cp_bin = ensure_same_shape(intensity_img, cp_bin)
            ws_mask = ensure_same_shape(intensity_img, ws_mask)

            # Run the detailed analysis
            detailed_df, processed_cells, total_cells = detailed_cell_analysis_three_way(
                full_image=intensity_img,
                gt_inst=gt_inst,
                gt_bin=gt_bin,
                cp_inst=cp_inst,
                cp_bin=cp_bin,
                ws_mask=ws_mask,
                patch_size=args.patch_size,
                base=base,
                debug_mode=args.debug,
                output_dir=args.output
            )

            # Add image identifier to detailed results
            detailed_df["Image"] = base
            detailed_results_list.append(detailed_df)

            # Calculate aggregated metrics
            aggregated = detailed_df.agg({
                'GT_area': 'mean',
                'CP_area': 'mean',
                'WS_area': 'mean',
                'GT_perimeter': 'mean',
                'CP_perimeter': 'mean',
                'WS_perimeter': 'mean',
            })
            
            # Store results
            overall_results.append({
                "Image": base,
                "GT_area": float(f"{aggregated['GT_area']:.1f}"),
                "CP_area": float(f"{aggregated['CP_area']:.1f}"),
                "WS_area": float(f"{aggregated['WS_area']:.1f}"),
                "GT_perimeter": float(f"{aggregated['GT_perimeter']:.1f}"),
                "CP_perimeter": float(f"{aggregated['CP_perimeter']:.1f}"),
                "WS_perimeter": float(f"{aggregated['WS_perimeter']:.1f}"),
                "Total_GT_Cells": total_cells,
                "Processed_Cells": processed_cells
            })

        except Exception as e:
            logging.error(f"Error processing {base}: {str(e)}")
            continue

    # Save overall results
    if overall_results:
        overall_df = pd.DataFrame(overall_results)
        overall_csv = os.path.join(args.output, 
            f"area_perimeter_aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        overall_df.to_csv(overall_csv, index=False)
        logging.info(f"Saved overall results to {overall_csv}")

    # Save detailed results
    if detailed_results_list:
        detailed_df_all = pd.concat(detailed_results_list, ignore_index=True)
        detailed_csv = os.path.join(args.output, 
            f"area_perimeter_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        detailed_df_all.to_csv(detailed_csv, index=False)
        logging.info(f"Saved detailed results to {detailed_csv}")

if __name__ == "__main__":
    main()