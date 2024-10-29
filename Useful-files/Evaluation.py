import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv
import os

def resize_data(data, target_shape):
    """
    Resizes the input data to the target shape.
    """
    factors = np.array(target_shape) / np.array(data.shape)
    resized_data = zoom(data, factors, order=1)  # Using bilinear interpolation (order=1)
    return resized_data

def calculate_mse_slice_by_slice(original_nifti_path, reconstructed_nifti_path, mask_nifti_path):
    """
    Calculates the mean squared error (MSE) between the original and reconstructed NIfTI volumes,
    considering only the voxels within the mask, slice by slice.
    """
    # Load the NIfTI files
    original_nifti = nib.load(original_nifti_path)
    reconstructed_nifti = nib.load(reconstructed_nifti_path)
    mask_nifti = nib.load(mask_nifti_path)

    # Get the data from the NIfTI files
    original_data = original_nifti.get_fdata()
    reconstructed_data = reconstructed_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    # Ensure data types are float32 for calculations
    original_data = original_data.astype('float32')
    reconstructed_data = reconstructed_data.astype('float32')
    mask_data = mask_data.astype('float32')
    #original_data = original_data.astype('uint8')
    #reconstructed_data = reconstructed_data.astype('uint8')
    #mask_data = mask_data.astype('uint8')

    # Normalize the original and reconstructed data based on fixed min/max values
    min_val = -1000
    #min_val = 0
    max_val = 3050
    #max_val = 255
    data_range = max_val - min_val

    # Resize the reconstructed data to match the original data's shape
    if original_data.shape != reconstructed_data.shape:
        reconstructed_data = resize_data(reconstructed_data, original_data.shape)

    # Initialize lists to store MSE, PSNR, SSIM, and weights for each slice
    mse_slices = []
    mae_slices = []
    psnr_slices = []
    ssim_slices = []
    weights = []
    total_weighted_mse = 0.0
    total_weighted_mae = 0.0
    total_weight = 0

    # Calculate the overall metrics using the mask
    original_masked = original_data[mask_data > 0]
    reconstructed_masked = reconstructed_data[mask_data > 0]
    mask_indices_global = np.where(mask_data > 0)
    num_masked_voxels = len(mask_indices_global[0])
    mse_overall = np.sum((original_masked - reconstructed_masked) ** 2) / num_masked_voxels
    mae_overall = (np.sum(np.abs(original_masked - reconstructed_masked)) / num_masked_voxels)
    #mse_overall = np.sum((original_masked.astype(np.int64) - reconstructed_masked.astype(np.int64)) ** 2) / num_masked_voxels
    #mae_overall = (np.sum(np.abs(original_masked.astype(np.int64) - reconstructed_masked.astype(np.int64))) / num_masked_voxels)
    psnr_overall = peak_signal_noise_ratio(original_masked, reconstructed_masked, data_range=data_range)
    ssim_overall = structural_similarity(original_masked, reconstructed_masked, multichannel=False, data_range=data_range)

    # Calculate slice-by-slice metrics
    for i in range(original_data.shape[2]):  # Assuming slices along the third dimension
        original_slice = original_data[:, :, i]
        reconstructed_slice = reconstructed_data[:, :, i]
        mask_slice = mask_data[:, :, i]

        mask_indices_slice = np.where(mask_slice > 0)
        original_masked_slice = original_slice[mask_indices_slice]
        reconstructed_masked_slice = reconstructed_slice[mask_indices_slice]

        # Calculate MSE, PSNR, SSIM for this slice
        num_masked_voxels_slice = len(mask_indices_slice[0])
        #mse_slice = np.sum((original_masked_slice.astype(np.int64) - reconstructed_masked_slice.astype(np.int64)) ** 2) / num_masked_voxels_slice
        #mae_slice = (np.sum(np.abs(original_masked_slice.astype(np.int64) - reconstructed_masked_slice.astype(np.int64))) / num_masked_voxels_slice)
        mse_slice = np.sum((original_masked_slice - reconstructed_masked_slice) ** 2) / num_masked_voxels_slice
        mae_slice = (np.sum(np.abs(original_masked_slice - reconstructed_masked_slice)) / num_masked_voxels_slice)
        psnr_slice = peak_signal_noise_ratio(original_masked_slice, reconstructed_masked_slice, data_range=data_range)
        ssim_slice = structural_similarity(original_masked_slice, reconstructed_masked_slice, multichannel=False, data_range=data_range)

        # Store the results
        mse_slices.append(mse_slice)
        mae_slices.append(mae_slice)
        psnr_slices.append(psnr_slice)
        ssim_slices.append(ssim_slice)
        weights.append(num_masked_voxels_slice)

        # Accumulate weighted MSE
        total_weighted_mse += mse_slice * num_masked_voxels_slice
        total_weighted_mae += mae_slice * num_masked_voxels_slice
        total_weight += num_masked_voxels_slice

    mse_weighted_average = total_weighted_mse / total_weight
    mae_weighted_average = total_weighted_mae / total_weight
    psnr_mean_slices = np.mean(psnr_slices)
    ssim_mean_slices = np.mean(ssim_slices)


    return mse_slices, mae_slices, psnr_slices, ssim_slices, mse_overall, mae_overall, psnr_overall, ssim_overall, mse_weighted_average, mae_weighted_average, psnr_mean_slices, ssim_mean_slices, weights, total_weight

def process_patients(patient_list, results_dir, output_dir):
    """
    Processes each patient in the list, calculates the metrics, and saves them to CSV files.
    """
    model_metrics = []

    for patient in patient_list:
        # Paths to the NIfTI files for this patient
        reconstructed_nifti_path = f'{results_dir}/{patient}_rescaled.nii.gz'
        original_nifti_path = f'D:/Datasets/Task2/brain/{patient}/ct_rescaled.nii.gz'
        mask_nifti_path = f'D:/Datasets/Task2/brain/{patient}/mask.nii.gz'

        # Calculate the metrics
        mse_slices, mae_slices, psnr_slices, ssim_slices, mse_overall, mae_overall, psnr_overall, ssim_overall, mse_weighted_average, mae_weighted_average, psnr_mean_slices, ssim_mean_slices, weights, total_weight = calculate_mse_slice_by_slice(
            original_nifti_path, reconstructed_nifti_path, mask_nifti_path)


        # Save slice-by-slice metrics to CSV for this patient
        csv_filename = os.path.join(output_dir, f'{patient}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['Slice', 'MSE', 'MAE', 'PSNR', 'SSIM', 'Weight'])
            for idx, (mse, mae, psnr, ssim, weight) in enumerate(zip(mse_slices, mae_slices, psnr_slices, ssim_slices, weights)):
                writer.writerow([idx, f'{mse:.6f}'.replace('.', ','), f'{mae:.6f}'.replace('.', ','), f'{psnr:.6f}'.replace('.', ','), f'{ssim:.6f}'.replace('.', ','), weight])
            # Add the final row with weighted average MSE, MAE, mean PSNR, mean SSIM, and total weight
            writer.writerow(['Total', f'{mse_weighted_average:.6f}'.replace('.', ','), f'{mae_weighted_average:.6f}'.replace('.', ','), f'{psnr_mean_slices:.6f}'.replace('.', ','), f'{ssim_mean_slices:.6f}'.replace('.', ','), total_weight])
        # Add overall metrics to the model CSV data
        model_metrics.append([patient, mse_overall, mae_overall, psnr_overall, ssim_overall])

    # Save overall metrics for all patients to a CSV file
    model_csv_filename = os.path.join(output_dir, 'modello.csv')
    with open(model_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Patient', 'MSE Overall', 'MAE Overall', 'PSNR Overall', 'SSIM Overall'])
        for row in model_metrics:
            writer.writerow([row[0], f'{row[1]:.6f}'.replace('.', ','), f'{row[2]:.6f}'.replace('.', ','), f'{row[3]:.6f}'.replace('.', ','), f'{row[4]:.6f}'.replace('.', ',')])
        # Calculate and write the average metrics across all patients
        mse_mean = np.mean([row[1] for row in model_metrics])
        mae_mean = np.mean([row[2] for row in model_metrics])
        psnr_mean = np.mean([row[3] for row in model_metrics])
        ssim_mean = np.mean([row[4] for row in model_metrics])

        # Calculate standard deviation for MSE, PSNR, and SSIM
        mse_std = np.std([row[1] for row in model_metrics])
        mae_std = np.std([row[2] for row in model_metrics])
        psnr_std = np.std([row[3] for row in model_metrics])
        ssim_std = np.std([row[4] for row in model_metrics])

        # Write average and standard deviation in the final row
        writer.writerow(['Average', f'{mse_mean:.6f} ± {mse_std:.6f}'.replace('.', ','),
                                   f'{mae_mean:.6f} ± {mae_std:.6f}'.replace('.', ','),
                                   f'{psnr_mean:.6f} ± {psnr_std:.6f}'.replace('.', ','),
                                   f'{ssim_mean:.6f} ± {ssim_std:.6f}'.replace('.', ',')])

if __name__ == '__main__':
    # List of patients
    patient_list = [
        '2BA002', '2BA023', '2BA030', '2BA080', '2BB030', '2BB034', '2BB071',
        '2BB079', '2BB096', '2BB100', '2BB102', '2BB145', '2BB189', '2BC002',
        '2BC016', '2BC045', '2BC047', '2BC090'
    ]

    #patient_list = [
        #'2PA002', '2PA016', '2PA030', '2PA078', '2PB027', '2PB036', '2PB068',
        #'2PB081', '2PB095', '2PB101', '2PB103', '2PB108', '2PB124', '2PC003',
        #'2PC014', '2PC041', '2PC044', '2PC098'
    #]

    # Directories
    #results_dir = 'D:/test_results_tiff16_fixed/synthrad_brain_cbct2ct'
    #results_dir = 'D:/Test_cyclegan_finale/synthrad_brain_cbct2ct'
    results_dir = 'D:/test_inpainting_8bit/test_latest'
    output_dir = 'D:/'

    # Process all patients and save results
    process_patients(patient_list, results_dir, output_dir)


