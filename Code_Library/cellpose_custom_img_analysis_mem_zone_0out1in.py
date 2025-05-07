from pathlib import Path
import os
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import exposure
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from cellpose import models
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
import time
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, binary_erosion, binary_dilation, disk
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from collections import defaultdict
from datetime import datetime

# Start measuring time
start_total_time = time.time()

# Function Definitions
def load_input_file(input_folder_path, input_file_name):
    return pd.read_excel(os.path.join(input_folder_path, input_file_name))

# def initialize_model():
#     model = models.Cellpose(gpu=False, model_type=None)
#     channels = [0, 0]
#     checkpoint_path = '/Users/mariannakhanivetskaya/Desktop/Img_AnalysisCode/CFTRimg\ -\ Copy\ 2/checkpoint.pth'
#     model.load_model(checkpoint_path)
#     return model, channels

# def initialize_model():
#     checkpoint_path = '/Users/mariannakhanivetskaya/Desktop/Img_AnalysisCode/CFTRimg - Copy 2/checkpoint.pth'
#     # Pass the custom checkpoint path as the model_type argument
#     model = models.Cellpose(gpu=False, model_type=checkpoint_path)
#     channels = [0, 0]
#     return model, channels

def initialize_model():
    checkpoint_path = '/Users/mariannakhanivetskaya/Desktop/Img_AnalysisCode/CFTRimg - Copy 2/checkpoint.pth'
    # Initialize with model_type=None
    model = models.Cellpose(gpu=False, model_type=None)
    # Access the internal neural network (net) through cp to load the checkpoint
    model.cp.net.load_model(checkpoint_path, device=model.device)
    channels = [0, 0]
    return model, channels

def initialize_plates_data():
    return defaultdict(lambda: defaultdict(lambda: {
        'cells': {},
        'metrics_per_well': {
            'red_entire_negative_count': 0,
            'yel_entire_negative_count': 0,
            'yel_membrane_negative_count': 0,
            'total_cells_with_data': 0
        }
    }))

def load_and_preprocess_image(image_path):
    image = tiff.imread(image_path)
    return exposure.equalize_adapthist(image, clip_limit=0.03)

def segment_image(model, image, channels):
    start_time = time.time()
    masks, _, _, _ = model.eval(
        image,
        diameter=47.236,
        channels=channels,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        normalize=True,
        do_3D=False,
        resample=True,
        min_size=15,
        compute_masks=True,
        progress=None
    )
    end_time = time.time()
    print(f"Segmentation took {(end_time - start_time) / 60} minutes")
    return clear_border(masks)

def visualize_segmentation(image, masks):
    if image.dtype != 'uint8':
        image = img_as_ubyte(image)
    
    labeled_image = label2rgb(masks, image=image, bg_label=0, alpha=0.4)
    boundaries = find_boundaries(masks, mode='outer')
    dilated_boundaries = dilation(boundaries, disk(3))

    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(image, cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    plt.show()

    labeled_image_rgb = labeled_image.copy()
    labeled_image_rgb[boundaries] = [1, 1, 1]
    
    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(labeled_image_rgb)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def process_cell_metrics(masks_cleared, base_folder, plate_str, file_prefix, well_name, condition, norm_condition, plates_data):
    pixel_added = 5
    
    # Parameters for membrane definition - no outer extension, only inner pixels
    membrane_inner = 1  # We want 1 pixel inward from boundary
    
    for cell_label in np.unique(masks_cleared)[1:]:
        cell_mask = masks_cleared == cell_label
        properties = regionprops(cell_mask.astype(int))

        if properties:
            cell_area = properties[0].area
            cell_major_axis_length = properties[0].major_axis_length
            pixel_size_um = 0.3

            if 108 <= (cell_area * (pixel_size_um ** 2)) <= 5400 and (cell_major_axis_length * pixel_size_um) <= 32.4:
                skip_cell = False
                temp_cell_data = {
                    'red_entire': {t: None for t in range(1, 34)},
                    'yel_entire': {t: None for t in range(1, 34)},
                    'yel_membrane': {t: None for t in range(1, 34)},
                    'condition': condition,
                    'norm_condition': norm_condition
                }

                for time_point in range(1, 34):
                    image_path_red = os.path.join(base_folder, plate_str, "Images", f"{well_name}{file_prefix}ch1sk{time_point}fk1fl1.tiff")
                    image_path_yel = os.path.join(base_folder, plate_str, "Images", f"{well_name}{file_prefix}ch2sk{time_point}fk1fl1.tiff")

                    image_yel = tiff.imread(image_path_yel)
                    image_red = tiff.imread(image_path_red)

                    background_mask = ~masks_cleared.astype(bool)
                    Ir_background_avg = np.mean(image_red[background_mask])
                    Iy_background_avg = np.mean(image_yel[background_mask])

                    minr, minc, maxr, maxc = properties[0].bbox
                    minr = max(minr - pixel_added, 0)
                    minc = max(minc - pixel_added, 0)
                    maxr = min(maxr + pixel_added, cell_mask.shape[0])
                    maxc = min(maxc + pixel_added, cell_mask.shape[1])
                    
                    cropped_mask = cell_mask[minr:maxr, minc:maxc]
                    
                    # Create a mask that's eroded by membrane_inner pixels
                    eroded_mask = binary_erosion(cropped_mask, disk(membrane_inner))
                    
                    # Create a mask that's eroded by membrane_inner + 1 pixels
                    deeper_eroded_mask = binary_erosion(cropped_mask, disk(membrane_inner + 1))
                    
                    # Membrane mask includes boundary pixels plus membrane_inner pixels inward
                    # Get the boundary pixels
                    boundary = cropped_mask & ~eroded_mask
                    # Get the pixels membrane_inner pixels inward from boundary
                    inner_ring = eroded_mask & ~deeper_eroded_mask
                    # Combine to get final membrane mask
                    membrane_mask = boundary | inner_ring

                    cropped_yellow = image_yel[minr:maxr, minc:maxc]
                    cropped_red = image_red[minr:maxr, minc:maxc]

                    red_crop_adj = cropped_red - Ir_background_avg
                    yel_crop_adj = cropped_yellow - Iy_background_avg

                    entire_pixel_count = np.sum(cropped_mask)
                    membrane_pixel_count = np.sum(membrane_mask)

                    red_entire = np.sum(cropped_mask * red_crop_adj) / entire_pixel_count
                    yel_entire = np.sum(cropped_mask * yel_crop_adj) / entire_pixel_count
                    yel_membrane = np.sum(membrane_mask * yel_crop_adj) / membrane_pixel_count

                    # Check if any fluorescence densities are negative and increment counters
                    if np.sum(red_entire) < 0:
                        plates_data[plate_str][well_name]['metrics_per_well']['red_entire_negative_count'] += 1
                        skip_cell = True
                    if np.sum(yel_entire) < 0:
                        plates_data[plate_str][well_name]['metrics_per_well']['yel_entire_negative_count'] += 1
                        skip_cell = True
                    if np.sum(yel_membrane) < 0:
                        plates_data[plate_str][well_name]['metrics_per_well']['yel_membrane_negative_count'] += 1
                        skip_cell = True

                    # If negative, set the flag to skip the rest of processing for this cell
                    if skip_cell:
                        break

                    temp_cell_data['red_entire'][time_point] = red_entire
                    temp_cell_data['yel_entire'][time_point] = yel_entire
                    temp_cell_data['yel_membrane'][time_point] = yel_membrane

                if not skip_cell:
                    plates_data[plate_str][well_name]['cells'][cell_label] = temp_cell_data
                    plates_data[plate_str][well_name]['metrics_per_well']['total_cells_with_data'] += 1
            
                    if cell_label not in plates_data[plate_str][well_name]['cells']:
                        plates_data[plate_str][well_name]['cells'][cell_label] = {
                            'red_entire': {t: None for t in range(1, 34)},
                            'yel_entire': {t: None for t in range(1, 34)},
                            'yel_membrane': {t: None for t in range(1, 34)},
                            'condition': condition,
                            'norm_condition': norm_condition
                        }

def normalize_metrics(plates_data):
    for plate_name, wells in plates_data.items():
        for well_name, well_data in wells.items():
            for cell_label, cell_metrics in well_data['cells'].items():
                red_reference = cell_metrics['red_entire'].get(12, None)
                yel_reference = cell_metrics['yel_entire'].get(12, None)

                cell_metrics['red_entire_to_norm'] = {t: None for t in range(1, 34)}
                cell_metrics['yel_entire_to_norm'] = {t: None for t in range(1, 34)}

                if red_reference is not None and red_reference != 0:
                    for time_point in range(1, 34):
                        red_value = cell_metrics['red_entire'].get(time_point, None)
                        if red_value is not None:
                            cell_metrics['red_entire_to_norm'][time_point] = red_value / red_reference

                if yel_reference is not None and yel_reference != 0:
                    for time_point in range(1, 34):
                        yel_value = cell_metrics['yel_entire'].get(time_point, None)
                        if yel_value is not None:
                            cell_metrics['yel_entire_to_norm'][time_point] = yel_value / yel_reference

def calculate_auc(y_values, x_values):
    return np.trapz(y_values, x=x_values)

def save_results_to_excel(df, plate_averages, save_folder):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    file_name = f"full_quench_results_{timestamp}.xlsx"
    file_path = os.path.join(save_folder, file_name)

    with pd.ExcelWriter(file_path) as writer:
        df.to_excel(writer, sheet_name='full_results', index=False)
        plate_averages.to_excel(writer, sheet_name='plate_averages', index=False)

    print(f"Analysis results saved to {file_path}")

def process_well(row, model, channels, plates_data):
    save_results_here = row['saveResultsHere']
    plate_str = row['plateStr']
    base_folder = row['baseFolder']
    file_prefix = row['filePrefix']
    condition = row['condition']
    norm_condition = row['normCondition']

    well_columns = [col for col in row.index if col.startswith('condWells')]

    for well in well_columns:
        well_name = row[well]
        if pd.isna(well_name) or well_name == 'nan':
            continue
        print(f"Now we are analysing well {well_name}")

        time_point = 12 # need to be replaced; hardcoded
        image_path_red = os.path.join(base_folder, plate_str, "Images", f"{well_name}{file_prefix}ch1sk{time_point}fk1fl1.tiff")
        image_red_eq = load_and_preprocess_image(image_path_red)
        masks_cleared = segment_image(model, image_red_eq, channels)
        # visualize_segmentation(image_red_eq, masks_cleared)
        process_cell_metrics(masks_cleared, base_folder, plate_str, file_prefix, well_name, condition, norm_condition, plates_data)

def create_dataframe_from_plates_data(plates_data):
    data_rows = []
    time_points = [0] + list(range(3, 44, 2))
    for plate_name, wells in plates_data.items():
        for well_name, well_data in wells.items():
            for cell_label, cell_metrics in well_data['cells'].items():
                row = {
                    'plate_name': plate_name,
                    'well': well_name,
                    'condition': cell_metrics['condition'],
                    'cell_label': cell_label,
                    'norm_condition': cell_metrics['norm_condition'],
                    'red_entire_12': cell_metrics['red_entire'].get(12, None),
                    'yel_entire_12': cell_metrics['yel_entire'].get(12, None),
                    'yel_membrane_12': cell_metrics['yel_membrane'].get(12, None),
                    'red_entire_negative_count': well_data['metrics_per_well']['red_entire_negative_count'],
                    'yel_entire_negative_count': well_data['metrics_per_well']['yel_entire_negative_count'],
                    'yel_membrane_negative_count': well_data['metrics_per_well']['yel_membrane_negative_count'],
                    'total_cells_with_data': well_data['metrics_per_well']['total_cells_with_data']
                }

                # Add red_entire_to_norm values with renamed columns
                for i, tp in enumerate(range(12, 34)):
                    norm_value = cell_metrics['red_entire_to_norm'].get(tp, None)
                    norm_col = f"r_{time_points[i]}"
                    row[norm_col] = norm_value
    
                # Add yel_entire_to_norm values with renamed columns
                for i, tp in enumerate(range(12, 34)):
                    norm_value = cell_metrics['yel_entire_to_norm'].get(tp, None)
                    norm_col = f"y_{time_points[i]}"
                    row[norm_col] = norm_value
    
                data_rows.append(row)

    return pd.DataFrame(data_rows)

def localisation_normalisation(df):
    numeric_columns = df.select_dtypes(include='number').columns
    medians = df.groupby('norm_condition')[numeric_columns].median()[['yel_entire_12', 'red_entire_12']]

    df = df.merge(medians, on='norm_condition', suffixes=('', '_median'))

    df['yel_entire_12_norm'] = df['yel_entire_12'] / df['yel_entire_12_median']
    df['red_entire_12_norm'] = df['red_entire_12'] / df['red_entire_12_median']
    df['yel_membrane_12_norm'] = df['yel_membrane_12'] / df['yel_entire_12_median']

    df['memDens'] = df['yel_membrane_12_norm'] / df['red_entire_12_norm']
    df['yelEntireN_redEntireN'] = df['yel_entire_12_norm'] / df['red_entire_12_norm']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['logMemDens'] = np.log10(df['memDens'].replace(0, np.nan).clip(lower=1e-10))
    df['log10_yelEntireN_redEntireN'] = np.log10(df['yelEntireN_redEntireN'].replace(0, np.nan).clip(lower=1e-10))

    df.drop(['yel_entire_12_median', 'red_entire_12_median'], axis=1, inplace=True)

    df.fillna(0, inplace=True)
    time_points = [0] + list(range(3, 44, 2))
    for cell_idx, cell_data in df.iterrows():
        y_values = [1] + [cell_data.get(f'y_{tp}', np.nan) for tp in time_points[1:]]
        for i, tp in enumerate(time_points[1:], start=1):
            valid_indices = ~np.isnan(y_values[:i+1])
            auc = calculate_auc(np.array(y_values[:i+1])[valid_indices], np.array(time_points[:i+1])[valid_indices])
            aac = tp - auc
            df.at[cell_idx, f'AAC_{tp}'] = aac

    return df

def calculate_plate_averages(df):
    time_points = [0] + list(range(3, 44, 2))
    columns = [
        'plate_name', 'condition', 'mean_logMemDens', 'mean_log10_yelEntireN_redEntireN',
        'mean_memDens', 'mean_yelEntireN_redEntireN', 'mean_red_entire_norm'
    ] + [f'mean_AAC_{tp}' for tp in time_points[1:]] + [f'mean_y_{tp}' for tp in time_points] + [f'mean_r_{tp}' for tp in time_points]

    plate_averages = pd.DataFrame(columns=columns)

    grouped = df.groupby(['plate_name', 'condition'])

    for (plate, condition), group in grouped:
        mean_log_mem_dens = group['logMemDens'].mean()
        mean_log10_yel_entire_n_red_entire_n = group['log10_yelEntireN_redEntireN'].mean()
        mean_mem_dens = group['memDens'].mean()
        mean_yelEntireN_redEntireN = group['yelEntireN_redEntireN'].mean()
        mean_red_entire_norm = group['red_entire_12_norm'].mean()
        
        mean_aac = {f'mean_AAC_{tp}': group[f'AAC_{tp}'].mean() for tp in time_points[1:]}
        mean_y = {f'mean_y_{tp}': group[f'y_{tp}'].mean() for tp in time_points}
        mean_r = {f'mean_r_{tp}': group[f'r_{tp}'].mean() for tp in time_points}
        
        row = {
            'plate_name': plate,
            'condition': condition,
            'mean_logMemDens': mean_log_mem_dens,
            'mean_log10_yelEntireN_redEntireN': mean_log10_yel_entire_n_red_entire_n,
            'mean_memDens': mean_mem_dens,
            'mean_yelEntireN_redEntireN': mean_yelEntireN_redEntireN,
            'mean_red_entire_norm': mean_red_entire_norm,
            **mean_aac,
            **mean_y,
            **mean_r
        }
        
        plate_averages = pd.concat([plate_averages, pd.DataFrame([row])], ignore_index=True)

    return plate_averages

# Main execution
input_folder_path = Path("/Users/mariannakhanivetskaya/Desktop/Img_AnalysisCode/CFTRimg - Copy 2")
input_file_name = "Inputfile_python_opera_phenix_Marianna_Jyosthnas_GoSlo_acute_p2.xlsx"
input_df = load_input_file(input_folder_path, input_file_name)

model, channels = initialize_model()
plates_data = initialize_plates_data()

for idx, row in input_df.iterrows():
    process_well(row, model, channels, plates_data)

normalize_metrics(plates_data)

df = create_dataframe_from_plates_data(plates_data)
df = localisation_normalisation(df)
plate_averages = calculate_plate_averages(df)

save_results_here = input_df['saveResultsHere'].iloc[0]
save_results_to_excel(df, plate_averages, save_results_here)

# End measuring time and print total time taken
end_total_time = time.time()
total_time_taken = end_total_time - start_total_time
print(f"Total time taken: {total_time_taken:.2f} seconds")