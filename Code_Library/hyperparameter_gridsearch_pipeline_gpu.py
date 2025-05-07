#!/usr/bin/env python3
"""
Patch‑Based Hyperparameter Grid‑Search for Custom Cellpose Model Training

This script:
  1. Loads training images (TIFF) from the specified folder and, for each image, constructs the corresponding instance mask filename
     by appending '_cp_masks.npy' to the image base name. These seg.npy files (exported from the Cellpose GUI)
     contain instance masks with unique integer labels for each cell.
  2. Enhances training images using adaptive histogram equalization.
  3. Extracts random patches (size = 1/3 of image dimensions) from each training image.
     ~1000 patches are generated in total (e.g., ~250 per image if 4 images),
     then 900 are used for training and 100 for validation.
  4. Writes these training patches to a temporary directory and pauses for manual inspection.
  5. For each hyperparameter combination (learning rate and epochs), the training patches
     are written to a temporary directory (optionally copying in saved flow files) and the Cellpose CLI is invoked.
  6. The trained model is loaded and evaluated on the validation patches using Cellpose's average_precision metric.
  7. The best hyperparameters are selected.
  8. A final model is retrained on all 1000 patches using the best hyperparameters.
  9. A single random patch from a test image is extracted and used for evaluation.
 10. The final model weights are saved to the specified output directory.

Additional debugging:
  - Saves the instance masks loaded from .npy files into a specified debug folder.
  - Saves a few example patches (both image and mask) for manual inspection.
  - Logs unique label values for both training and test masks.

Usage example:
    python patch_based_gridsearch_custom.py --train_img_dir data/train/images/ \
         --train_mask_dir data/train/masks/ --test_img_path data/test/image5.tif \
         --test_mask_path data/test/r07c11-ch1-2_cp_masks.npy --custom_model_path models/my_custom_model \
         --output_dir results/ --visualize_patches --reuse_flows --flows_dir /path/to/saved/flows

Dependencies:
    - cellpose, numpy, tifffile, scikit-image, scikit-learn, argparse, logging,
      glob, os, random, subprocess, tempfile, shutil, cv2, matplotlib, tqdm, re
"""

import os
import glob
import argparse
import logging
import itertools
import random
import pickle
import numpy as np
import tifffile as tiff
import subprocess
import tempfile
import shutil
from skimage import exposure
from skimage.io import imread, imsave
from cellpose import models
from cellpose.metrics import average_precision
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import torch
import atexit

# ===== SET TMPDIR TO EXTERNAL DRIVE =====
external_temp = "/tmp/jarvis_temp"
if not os.path.exists(external_temp):
    os.makedirs(external_temp)
tempfile.tempdir = external_temp
print("Temporary directory is set to:", tempfile.gettempdir())
# ========================================

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', filename='/root/training_logs.log')

# -----------------------------
# Checkpointing Functions
# -----------------------------
CHECKPOINT_FILE = "checkpoint.pkl"

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(state, f)
    logging.info("Checkpoint saved.")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            state = pickle.load(f)
        logging.info("Checkpoint loaded.")
        return state
    return {}

# -----------------------------
# Recursive Extraction Helper
# -----------------------------
def recursive_extract(data):
    """Recursively search for the first NumPy array (with a non-object dtype) in data."""
    if isinstance(data, np.ndarray) and data.dtype != np.dtype('O'):
        return data
    if isinstance(data, dict):
        if 'masks' in data:
            return recursive_extract(data['masks'])
        for key, value in data.items():
            result = recursive_extract(value)
            if result is not None:
                return result
    if isinstance(data, list):
        for item in data:
            result = recursive_extract(item)
            if result is not None:
                return result
    return None

# -----------------------------
# Debugging Functions
# -----------------------------
def debug_print_structure(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        logging.info(f"Directory: {dirpath}")
        for fname in filenames:
            logging.info(f"  File: {fname}")

def debug_list_in_folders(temp_train_dir):
    subdirs = [d for d in os.listdir(temp_train_dir) if os.path.isdir(os.path.join(temp_train_dir, d))]
    logging.info(f"Subdirectories in {temp_train_dir}: {subdirs}")
    for sub in subdirs:
        sub_path = os.path.join(temp_train_dir, sub)
        debug_print_structure(sub_path)

def visualize_patch(patch_img, patch_mask, full_mask, patch_coords, patch_id, source_info="Ground Truth Mask", img_index=None):
    if full_mask is None:
        logging.error(f"Full mask is None for patch {patch_id}. Check file path and naming!")
        return
    x, y, w, h = patch_coords
    thickness = 10
    if len(full_mask.shape) == 2:
        full_mask_bgr = cv2.cvtColor(full_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        full_mask_bgr = full_mask.copy()
    cv2.rectangle(full_mask_bgr, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    full_mask_vis = cv2.cvtColor(full_mask_bgr, cv2.COLOR_BGR2RGB)
    if len(patch_img.shape) == 2:
        patch_img_vis = cv2.cvtColor((patch_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif patch_img.shape[2] == 1:
        patch_img_vis = cv2.cvtColor((patch_img[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        patch_img_vis = cv2.cvtColor((patch_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    patch_img_vis = patch_img_vis.astype(np.float32)
    patch_img_vis[..., 0] *= 1.3
    patch_img_vis = np.clip(patch_img_vis, 0, 255).astype(np.uint8)
    patch_mask_vis = cv2.cvtColor(((patch_mask > 0) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(patch_img_vis)
    axes[0].set_title("Raw Patch Image", fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(patch_mask_vis)
    axes[1].set_title("Patch Instance Mask", fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(full_mask_vis)
    axes[2].set_title("Full Ground Truth with Patch Highlight", fontsize=14)
    axes[2].axis('off')
    overall_title = f"Image {img_index} - {patch_id}: {source_info}" if img_index is not None else f"{patch_id}: {source_info}"
    fig.suptitle(overall_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"debug_patch_{patch_id}_img{img_index}.png" if img_index is not None else f"debug_patch_{patch_id}.png"
    plt.savefig(out_path)
    logging.info(f"Saved debug visualization to {out_path}")
    plt.show()

# -----------------------------
# GPU Availability Check
# -----------------------------
# import torch
# if torch.backends.mps.is_available():
#     print("MPS is available! M2 GPU will be used.")
# else:
#     print("MPS is not available; falling back to CPU.")
# input("Press Enter to continue...")

# ---------------------------------------------------------
# Image Enhancement Function (for training images)
# ---------------------------------------------------------
def enhance_image(img):
    return exposure.equalize_adapthist(img, clip_limit=0.03)

# ---------------------------------------------------------
# Data Loading and Preprocessing (Instance Masks)
# ---------------------------------------------------------
def load_images_and_instance_masks(img_dir, mask_dir):
    """
    For each image in img_dir, load the image and construct the corresponding instance mask filename
    by appending '_cp_masks.npy' to the image base name.
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.tif')) + glob.glob(os.path.join(img_dir, '*.tiff')))
    images = []
    instance_masks = []
    base_names = []
    for p in img_paths:
        base_name = Path(p).stem  # e.g., "r03c11-ch1-2"
        mask_path = os.path.join(mask_dir, base_name + "_cp_masks.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        img = cv2.imread(p, -1)
        img = enhance_image(img)
        images.append(img)
        seg_data = np.load(mask_path, allow_pickle=True)
        # Use 'masks' key if present.
        if isinstance(seg_data, dict) and 'masks' in seg_data:
            inst_mask = seg_data['masks']
        else:
            inst_mask = seg_data
        # If inst_mask is an object array, extract its content recursively.
        if isinstance(inst_mask, np.ndarray) and inst_mask.dtype == np.dtype('O'):
            if inst_mask.size == 1:
                candidate = recursive_extract(inst_mask.item())
            else:
                candidate = recursive_extract(inst_mask)
            if candidate is None:
                raise ValueError(f"Could not extract a valid instance mask from {mask_path}")
            inst_mask = candidate
        if not isinstance(inst_mask, np.ndarray):
            raise ValueError(f"Instance mask in {mask_path} is not a NumPy array")
        try:
            inst_mask = inst_mask.astype(np.int32)
        except Exception as e:
            raise ValueError(f"Error converting instance mask from {mask_path} to int32: {e}")
        instance_masks.append(inst_mask)
        base_names.append(base_name)
    logging.info(f"Loaded {len(images)} images and {len(instance_masks)} instance masks.")
    return images, instance_masks, base_names

# ---------------------------------------------------------
# Patch Extraction with Filtering (for images and instance masks)
# ---------------------------------------------------------
def extract_random_patches(image, instance_mask, patch_size, num_patches, min_positive=5, max_attempts_factor=10):
    img_patches = []
    mask_patches = []
    coords = []  # (x, y, width, height)
    img_H, img_W = image.shape[:2]
    patch_H, patch_W = patch_size
    if img_H < patch_H or img_W < patch_W:
        raise ValueError("Patch size is larger than image dimensions.")
    max_attempts = max_attempts_factor * num_patches
    attempts = 0
    while len(img_patches) < num_patches and attempts < max_attempts:
        y = random.randint(0, img_H - patch_H)
        x = random.randint(0, img_W - patch_W)
        patch_img = image[y:y+patch_H, x:x+patch_W]
        patch_mask = instance_mask[y:y+patch_H, x:x+patch_W]
        # Count distinct cell labels (excluding background 0)
        if (len(np.unique(patch_mask)) - 1) >= min_positive:
            img_patches.append(patch_img)
            mask_patches.append(patch_mask)
            coords.append((x, y, patch_W, patch_H))
        attempts += 1
    if len(img_patches) < num_patches:
        logging.warning(f"Only {len(img_patches)} patches extracted after {attempts} attempts; expected {num_patches}.")
    return img_patches, mask_patches, coords

# ---------------------------------------------------------
# Temporary Directory Writer for Training Data
# ---------------------------------------------------------
# Global tracking of temporary directories
temp_dirs = set()

def cleanup_temp_dirs():
    """Clean up all temporary directories on exit"""
    for d in temp_dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                logging.info(f"Cleaned up temporary directory: {d}")
            except Exception as e:
                logging.error(f"Failed to clean up directory {d}: {e}")

# Register cleanup function
atexit.register(cleanup_temp_dirs)

def write_patches_to_temp_dir(patches_info):
    temp_dir = tempfile.mkdtemp(prefix="cellpose_train_")
    temp_dirs.add(temp_dir)  # Track this directory
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        counters = {}
        for base_name, patch, mask, _ in tqdm(patches_info, desc="Writing patches", mininterval=1):
            counters[base_name] = counters.get(base_name, 0) + 1
            file_stem = f"{base_name}_patch{counters[base_name]:03d}"
            image_path = os.path.join(temp_dir, file_stem + ".tif")
            mask_png_path = os.path.join(temp_dir, file_stem + "_cp_masks.png")
            mask_npy_path = os.path.join(temp_dir, file_stem + "_cp_masks.npy")
            imsave(image_path, (patch * 255).astype(np.uint8))
            imsave(mask_png_path, mask)
            np.save(mask_npy_path, mask)
        debug_print_structure(temp_dir)
        return temp_dir
    except Exception as e:
        if temp_dir in temp_dirs:
            temp_dirs.remove(temp_dir)
        shutil.rmtree(temp_dir)
        raise e

# ---------------------------------------------------------
# Function to Copy Saved Flows (Optional)
# ---------------------------------------------------------
def copy_saved_flows(src_dir, dest_dir):
    import shutil
    flow_files = glob.glob(os.path.join(src_dir, "*_flows.tif"))
    if not flow_files:
        logging.warning("No flow files found in flows directory: " + src_dir)
    for f in flow_files:
        shutil.copy(f, dest_dir)
    logging.info("Copied saved flows from {} to {}".format(src_dir, dest_dir))

# ---------------------------------------------------------
# CLI-Based Training Function
# ---------------------------------------------------------
def train_model_cli(temp_train_dir, custom_model_path, lr, n_epochs, reuse_flows=False, flows_dir=None):
    debug_list_in_folders(temp_train_dir)
    if reuse_flows and flows_dir is not None:
        copy_saved_flows(flows_dir, temp_train_dir)
    models_dir = os.path.join(temp_train_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    cmd = [
         "python", "-m", "cellpose",
         "--train",
         "--use_gpu",
         "--dir", temp_train_dir,
         "--pretrained_model", custom_model_path if custom_model_path else "cyto",
         "--learning_rate", str(lr),
         "--n_epochs", str(n_epochs),
         "--chan", "0",
         "--chan2", "0",
         "--mask_filter", "_cp_masks",
         "--savedir", models_dir,
         "--model_name_out", "final_model.pth",
         "--verbose"
    ]
    logging.info("Running training command: " + " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
    process.stdout.close()
    retcode = process.wait()
    if retcode:
        raise subprocess.CalledProcessError(retcode, cmd)
    model_file = os.path.join(models_dir, "final_model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model file not found at {model_file}")
    return model_file

def load_trained_model(model_path):
    return models.CellposeModel(gpu=True, pretrained_model=model_path)

# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_model(model, patch_images, patch_masks):
    if hasattr(model, 'model'):
        model.model.to('cuda')
    else:
        logging.warning("The provided model does not appear to have an internal 'model' attribute.")
    
    ap_scores = []
    for idx, (img, true_mask) in enumerate(zip(patch_images, patch_masks), start=1):
        try:
            # Get the model output and log its type and length
            result = model.eval(img, channels=[0, 0])
            logging.debug("model.eval returned type %s", type(result))
            if isinstance(result, (list, tuple)):
                logging.debug("model.eval returned a tuple/list with length %s", len(result))
            else:
                logging.error("model.eval did not return a tuple or list, got %s", type(result))
                raise ValueError("Unexpected output format from model.eval()")
            
            # Unpack result based on its length
            if len(result) == 4:
                pred_masks, flows, styles, diams = result
            elif len(result) == 3:
                pred_masks, flows, diams = result
                styles = None
            else:
                raise ValueError(f"Unexpected number of outputs: {len(result)}")
            
            # Now log information about pred_masks safely
            logging.debug("Type of pred_masks: %s", type(pred_masks))
            try:
                logging.debug("Shape of pred_masks: %s", np.array(pred_masks).shape)
            except Exception as e:
                logging.error("Error converting pred_masks to array: %s", e)
            
            # Compute average precision
        #     ap = average_precision(true_mask, pred_masks)
        #     ap_scores.append(ap)
        #     logging.debug(f"Validation patch {idx}: Average Precision = {ap:.3f}")
        # except Exception as e:
        #     logging.error(f"Error during evaluation on patch {idx}: {e}")
        #     ap_scores.append(0)

            ap = average_precision(true_mask, pred_masks)
            logging.debug("Raw average_precision output: %s", ap)
            # If average_precision returns a tuple, e.g., (ap_value, other_metric)
            # if isinstance(ap, tuple):
            #     ap_value = ap[0]
            # else:
            #     ap_value = ap
            # ap_scores.append(ap_value)
            # logging.debug(f"Validation patch {idx}: Average Precision = {ap_value:.3f}")
            logging.debug("Raw average_precision output: %s", ap)
            # Extract the AP at the 0.5 IoU threshold:
            # ap[0] is a NumPy array of AP values for thresholds [0.5, 0.75, 0.9].
            # We take the first element for the 0.5 threshold.
            ap_value = ap[0][0] if isinstance(ap, tuple) else ap
            ap_scores.append(ap_value)
            logging.debug(f"Validation patch {idx}: AP@0.5 = {ap_value:.3f}")
        except Exception as e:
            logging.error(f"Error during evaluation on patch {idx}: {e}")
            ap_scores.append(0)
    return np.mean(ap_scores)


def load_raw_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# ---------------------------------------------------------
# Main Hyperparameter Tuning Workflow
# ---------------------------------------------------------
def main(args):
    # Optionally load checkpoint if resume flag is set.
    state = {}
    if args.resume:
        state = load_checkpoint()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Load training images and instance masks.
    # train_images, train_instance_masks, train_basenames = load_images_and_instance_masks(args.train_img_dir, args.train_mask_dir)
    
    # 1. Load training images and instance masks.
    if "images_loaded" in state:
        train_images, train_instance_masks, train_basenames = state["images_loaded"]
        logging.info("Resumed from checkpoint: loaded training images and masks.")
    else:
        train_images, train_instance_masks, train_basenames = load_images_and_instance_masks(args.train_img_dir, args.train_mask_dir)
        state["images_loaded"] = (train_images, train_instance_masks, train_basenames)
        save_checkpoint(state)

    # Debug: Print and save unique label values from training masks.
    for base_name, inst_mask in zip(train_basenames, train_instance_masks):
        unique_vals = np.unique(inst_mask)
        logging.info(f"Unique labels in training mask {base_name}: {unique_vals}")
    
    # Debug: Save all loaded instance masks to a specified folder.
    debug_folder = '/tmp/debug_instance_masks'
  # Change to desired path
    os.makedirs(debug_folder, exist_ok=True)
    for base_name, inst_mask in zip(train_basenames, train_instance_masks):
        debug_path = os.path.join(debug_folder, f"{base_name}_instance_mask.png")
        imsave(debug_path, inst_mask)
        logging.info(f"Saved instance mask for {base_name} to {debug_path}")
    
    # 2. Determine patch size (1/3 of image dimensions).
    img_H, img_W = train_images[0].shape[:2]
    patch_size = (img_H // 3, img_W // 3)
    logging.info(f"Patch size set to {patch_size} (1/3 of image dimensions).")
    
    # 3. Extract patches from each training image.
    # patches_info = []
    # for idx, (base_name, img, inst_mask) in enumerate(zip(train_basenames, train_images, train_instance_masks), start=1):
    #     img_patches, mask_patches, patch_coords = extract_random_patches(img, inst_mask, patch_size, args.num_patches_per_image)
    #     positive_counts = [len(np.unique(mp)) - 1 for mp in mask_patches]
    #     logging.info(f"For image {base_name} (Image {idx}): min distinct cells = {min(positive_counts)}, "
    #                  f"max distinct cells = {max(positive_counts)}, mean = {np.mean(positive_counts):.2f}")
    #     for mp, coords in zip(mask_patches, patch_coords):
    #         if (len(np.unique(mp)) - 1) < 5:
    #             logging.warning(f"Patch at {coords} from image {base_name} (Image {idx}) has fewer than 5 cells.")
    #     for p, m, c in zip(img_patches, mask_patches, patch_coords):
    #         patches_info.append((base_name, p, m, c))
    # 3. Extract patches from each training image.
    if "patches_info" in state:
        patches_info = state["patches_info"]
        logging.info("Resumed from checkpoint: patches already extracted.")
    else:
        patches_info = []
        for idx, (base_name, img, inst_mask) in enumerate(zip(train_basenames, train_images, train_instance_masks), start=1):
            img_patches, mask_patches, patch_coords = extract_random_patches(img, inst_mask, patch_size, args.num_patches_per_image)
            positive_counts = [len(np.unique(mp)) - 1 for mp in mask_patches]
            logging.info(f"For image {base_name} (Image {idx}): min distinct cells = {min(positive_counts)}, "
                         f"max distinct cells = {max(positive_counts)}, mean = {np.mean(positive_counts):.2f}")
            for p, m, c in zip(img_patches, mask_patches, patch_coords):
                patches_info.append((base_name, p, m, c))
        if args.visualize_patches:
            sample_patch_img = img_patches[0]
            sample_patch_mask = mask_patches[0]
            original_mask_path = os.path.join(args.train_mask_dir, train_basenames[idx-1] + "_cp_masks.npy")
            try:
                seg_data = np.load(original_mask_path, allow_pickle=True)
                full_mask_inst = None
                if isinstance(seg_data, dict) and 'masks' in seg_data:
                    full_mask_inst = seg_data['masks']
                else:
                    full_mask_inst = seg_data
                full_mask_inst = recursive_extract(full_mask_inst)
                if full_mask_inst is None:
                    raise ValueError("No valid instance mask extracted.")
                full_mask_inst = full_mask_inst.astype(np.int32)
            except Exception as e:
                logging.error(f"Could not load instance mask from {original_mask_path}: {e}")
                full_mask_inst = None
            visualize_patch(sample_patch_img, sample_patch_mask, full_mask_inst, patch_coords[0],
                            patch_id=base_name, source_info="Instance Ground Truth Mask", img_index=idx)
    state["patches_info"] = patches_info
    save_checkpoint(state)
    # --- Pause for manual inspection of patches ---
    exam_dir = write_patches_to_temp_dir([p for p in patches_info if p[0] == train_basenames[0]])
    # input(f"Patches have been written to {exam_dir}. Please examine them and press Enter to continue to grid search training...")
    
    # 4. Shuffle and split patches: 900 for training, 100 for validation.
    random.shuffle(patches_info)
    if len(patches_info) < 1000:
        logging.error("Not enough patches extracted. Increase --num_patches_per_image.")
        raise ValueError("Total patches less than 1000.")
    train_patches_info = patches_info[:900]
    val_patches_info = patches_info[900:1000]
    logging.info(f"Split into {len(train_patches_info)} training patches and {len(val_patches_info)} validation patches.")
    
    # 5. Hyperparameter grid search.
    learning_rates = [0.001, 0.01, 0.1]
    epochs_list = [100, 150, 200]
    best_hp = None
    best_val_ap = -np.inf
    logging.info("Starting hyperparameter grid search on patches...")
    if "grid_search_results" in state:
        grid_search_results = state["grid_search_results"]
    else:
        grid_search_results = {}
    for lr, n_epochs in itertools.product(learning_rates, epochs_list):
        hp_key = f"lr_{lr}_epochs_{n_epochs}"
        if hp_key in grid_search_results:
            logging.info(f"Skipping already tested combination: {hp_key}")
            val_ap = grid_search_results[hp_key]
        else:
            logging.info(f"Testing combination: LR = {lr}, Epochs = {n_epochs}")
            temp_train_dir = None
            try:
                temp_train_dir = write_patches_to_temp_dir(train_patches_info)
                model_ckpt = train_model_cli(temp_train_dir, args.custom_model_path, lr, n_epochs,
                                           reuse_flows=args.reuse_flows, flows_dir=args.flows_dir)
                model = load_trained_model(model_ckpt)
                val_ap = evaluate_model(model, [p[1] for p in val_patches_info], [p[2] for p in val_patches_info])
                logging.info(f"Combination (LR={lr}, Epochs={n_epochs}): Avg Validation Precision = {val_ap:.3f}")
                grid_search_results[hp_key] = val_ap
                state["grid_search_results"] = grid_search_results
                save_checkpoint(state)
            except Exception as e:
                logging.error(f"Error with combination LR={lr}, Epochs={n_epochs}: {e}")
                grid_search_results[hp_key] = 0
                state["grid_search_results"] = grid_search_results
                save_checkpoint(state)
                continue
            finally:
                if temp_train_dir and os.path.exists(temp_train_dir):
                    shutil.rmtree(temp_train_dir)
                    if temp_train_dir in temp_dirs:
                        temp_dirs.remove(temp_train_dir)
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_hp = (lr, n_epochs)

    if best_hp is None:
        logging.error("No valid hyperparameter combination found.")
        return
    logging.info(f"Best hyperparameters: LR = {best_hp[0]}, Epochs = {best_hp[1]} with Avg Precision = {best_val_ap:.3f}")
    
    # 6. Retrain final model on all 1000 patches using best hyperparameters.
    final_temp_train_dir = write_patches_to_temp_dir(patches_info[:1000])
    final_model_ckpt = train_model_cli(final_temp_train_dir, args.custom_model_path, best_hp[0], best_hp[1],
                                       reuse_flows=args.reuse_flows, flows_dir=args.flows_dir)
    final_model = load_trained_model(final_model_ckpt)
    # Evaluate final model on the training set (first 900 patches)
    final_train_ap = evaluate_model(final_model, [p[1] for p in train_patches_info], [p[2] for p in train_patches_info])
    logging.info(f"Final Model Training Set Average Precision: {final_train_ap:.3f}")
    # Evaluate final model on the validation set (patches 900 to 1000)
    final_val_ap = evaluate_model(final_model, [p[1] for p in val_patches_info], [p[2] for p in val_patches_info])
    logging.info(f"Final Model Validation Set Average Precision: {final_val_ap:.3f}")
    
    # 7. Evaluate final model on one random patch from the test image.
    test_image = tiff.imread(args.test_img_path)
    test_image = enhance_image(test_image)

    if args.test_mask_path.endswith('.npy'):
        seg_data = np.load(args.test_mask_path, allow_pickle=True)
        logging.info(f"Loaded test seg_data of type: {type(seg_data)}")
        # If seg_data is a dict and has a 'masks' key, use it.
        if isinstance(seg_data, dict) and 'masks' in seg_data:
            test_mask_candidate = seg_data['masks']
            logging.info("Using 'masks' key from test seg_data.")
        else:
            test_mask_candidate = seg_data
            logging.info("Test seg_data does not have a 'masks' key; using entire seg_data.")
        
        # If the candidate is an object array, perform recursive extraction.
        if isinstance(test_mask_candidate, np.ndarray) and test_mask_candidate.dtype == np.dtype('O'):
            logging.info(f"Test mask candidate is an object array with size {test_mask_candidate.size}.")
            if test_mask_candidate.size == 1:
                test_mask = recursive_extract(test_mask_candidate.item())
                logging.info("Applied recursive extraction on test_mask_candidate.item().")
            else:
                test_mask = recursive_extract(test_mask_candidate)
                logging.info("Applied recursive extraction on test_mask_candidate array.")
        else:
            test_mask = test_mask_candidate

        if test_mask is None:
            raise ValueError(f"Could not extract a valid instance mask from {args.test_mask_path}")
        test_mask = np.array(test_mask).astype(np.int32)
        logging.info(f"Test mask unique labels: {np.unique(test_mask)}")
        # Re-label ground truth test mask sequentially starting from 0 and print new labels
        from skimage.segmentation import relabel_sequential
        new_test_mask, fw, inv = relabel_sequential(test_mask)
        logging.info("Unique labels in ground truth test mask after sequential relabeling: %s", np.unique(new_test_mask))
        test_mask = new_test_mask
    else:
        test_mask = (tiff.imread(args.test_mask_path) > 0).astype(np.uint8)

    if test_image.shape[0] < patch_size[0] or test_image.shape[1] < patch_size[1]:
        logging.error("Test image is smaller than required patch size.")
        raise ValueError("Test image too small.")

    test_img_patches, test_mask_patches, _ = extract_random_patches(test_image, test_mask, patch_size, 1)
    test_patch = test_img_patches[0]
    test_patch_mask = test_mask_patches[0]

    # Save the test patch for inspection:
    save_path = os.path.join(tempfile.gettempdir(), "test_patch.tif")
    imsave(save_path, (test_patch * 255).astype(np.uint8))
    logging.info("Test patch saved to: %s", save_path)

     # Evaluate the test patch once more to capture the predicted mask for inspection.
    try:
        eval_result = final_model.eval(test_patch, channels=[0, 0])
        if isinstance(eval_result, (list, tuple)):
            if len(eval_result) == 4:
                pred_masks, flows, styles, diams = eval_result
            elif len(eval_result) == 3:
                pred_masks, flows, diams = eval_result
                styles = None
            else:
                raise ValueError(f"Unexpected number of outputs: {len(eval_result)}")
        else:
            raise ValueError("model.eval() did not return a tuple or list.")
        # Save predicted mask for visual inspection.
        out_pred_path = os.path.join(tempfile.gettempdir(), "predicted_mask.tif")
        imsave(out_pred_path, pred_masks.astype(np.uint16))
        logging.info("Predicted mask saved to: %s", out_pred_path)
    except Exception as e:
        logging.error("Error during evaluation and saving predicted mask: %s", e)

    # Calculate Average Precision
    test_ap = evaluate_model(final_model, [test_patch], [test_patch_mask])
    logging.info(f"Final Model Test Average Precision on one patch: {test_ap:.3f}")

    # 8. Get best hyperparameter combination
    print("\nFinal Summary:")
    print("Best Hyperparameters: Learning Rate = {}, Epochs = {}".format(best_hp[0], best_hp[1]))

    # 9. Save the final model weights.
    os.makedirs(args.output_dir, exist_ok=True)
    final_save_path = os.path.join(args.output_dir, "final_custom_model_patch_weights.pth")
    try:
        torch.save(final_model.net.state_dict(), final_save_path)
        logging.info(f"Final model saved to {final_save_path}")
    except Exception as e:
        logging.error(f"Error saving final model: {e}")

# ---------------------------------------------------------
# Argument Parsing and Entry Point
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch‑Based Hyperparameter Grid Search for Custom Cellpose Model")
    parser.add_argument("--train_img_dir", type=str, default="/root/patch_based_gridsearch/data/train/images/",
                        help="Directory containing training images (TIFF).")
    parser.add_argument("--train_mask_dir", type=str, default="/root/patch_based_gridsearch/data/train/masks/",
                        help="Directory containing the corresponding instance mask files. (Expected naming: <base>_cp_masks.npy)")
    parser.add_argument("--test_img_path", type=str, default='/root/patch_based_gridsearch/data/test/r07c11-ch1-2.tif',
                        help="Path to the test image (TIFF).")
    parser.add_argument("--test_mask_path", type=str, default='/root/patch_based_gridsearch/data/test/r07c11-ch1-2_cp_masks.npy',
                        help="Path to the test instance mask (npy file).")
    parser.add_argument("--custom_model_path", type=str, default="/root/patch_based_gridsearch/models/cyto3_iterative7_final.pth",
                        help="Path to your custom model checkpoint (e.g., models/my_custom_model).")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save the final model weights.")
    parser.add_argument("--num_patches_per_image", type=int, default=275,
                        help="Number of patches to extract from each training image.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--visualize_patches", action="store_true",
                        help="If set, visualize one sample patch from each training image.")
    parser.add_argument("--reuse_flows", action="store_true",
                        help="If set, reuse saved flow files from a previous training run.")
    parser.add_argument("--flows_dir", type=str, default=None,
                        help="Directory containing saved flow files (with _flows.tif) to reuse.")
    parser.add_argument("--resume", action="store_true",
                        help="If set, resume from the last saved checkpoint.")
    
    args = parser.parse_args()
    main(args)

    import logging
    logging.shutdown()

