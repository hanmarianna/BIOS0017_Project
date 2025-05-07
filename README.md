# MSci BIOS0017 Project

This repository contains resources, scripts, and image data for the MSci BIOS0017 project.

## Repository Structure

- **Code_Library**
  - Contains scripts, pipelines, and configuration files used in image analysis and segmentation.
  - Files include:
    - `inputfile_python_opera_phenix.xlsx`: Excel file for input parameters.
    - `cellpose_cyto3_img_analysis_mem_zone_0out1in.py`: OperaPhenixPlus pIRES2-mCherry-YFP CFTR fluorescence assay image analysis pipeline using Cellpose (cyto3) segmentation model.
    - `cellpose_custom_img_analysis_mem_zone_0out1in.py`: OperaPhenixPlus pIRES2-mCherry-YFP CFTR fluorescence assay image analysis pipeline using Custom-trained Cellpose segmentation model.
    - `morphological_analysis_pipeline.py`: Pipeline for morphological analysis.
    - `hyperparameter_gridsearch_pipeline_gpu.py`: GPU-based hyperparameter grid search pipeline.
    - `cellpose_env.yaml`: Environment configuration file (e.g., for Conda).
    - `custom_cellpose_model.pth`: Model file.

- **Image_Library**
  - Stores image data and associated files for analysis.
  - Folders include:
    - `test`: Directory containing test images.
    - `human_in_the_loop`: Directory for images where human-in-the-loop adjustments are applied.
    - `ground_truth`: Directory containing ground truth image data.
  - Also includes some system files like `.DS_Store` that may appear automatically on macOS.

## Getting Started

1. Ensure your environment is set up using the configuration in `cellpose_env.yaml`.
2. Run the analysis scripts from the **Code_Library** as needed.
3. Refer to the **Image_Library** for the corresponding images used in analysis.

## Note

This README provides an overview of the repository structure. For further details on each component, please refer to the scripts and additional documentation within the repository. 
