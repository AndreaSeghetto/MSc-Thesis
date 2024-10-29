import os
import numpy as np
import nibabel as nib
import tifffile as tiff
from PIL import Image


def tiff_to_nifti(patient_id, tiff_dir, output_nifti_path):
    """
    Converts TIFF slices of a patient into a NIfTI volume.

    Args:
    - patient_id: ID of the patient.
    - tiff_dir: Directory containing TIFF slices of the patient.
    - output_nifti_path: Path to save the output NIfTI file.
    """
    # List all TIFF files in the directory that start with patient_id and end with '_ct.tif'
    tiff_files = sorted([os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if
                         f.startswith(f'{patient_id}') and f.endswith('_fake_B.tiff')])
                         #f.startswith(f'{patient_id}')])
                         #f.startswith(f'{patient_id}') and f.endswith('_fake_B.png')])

    num_slices = len(tiff_files)

    # Read one slice to get dimensions
    sample_slice = tiff.imread(tiff_files[0])
    sample_slice = sample_slice[0, 0, :, :]  # Select the last two dimensions
    #sample_slice = Image.open(tiff_files[0])
    slice_array = np.array(sample_slice)
    slice_shape = slice_array.shape
    slice_dtype = slice_array.dtype


    # Initialize a 3D array to hold all slices
    volume_data = np.zeros((slice_shape[0], slice_shape[1], num_slices), dtype=slice_dtype)


    # Read all TIFF slices into the volume array
    for idx, tiff_file in enumerate(tiff_files):
        #slice_data = np.array(Image.open(tiff_file).convert('L'))
        slice_data = tiff.imread(tiff_file)
        slice_data = slice_data[0, 0, :, :]  # Select the last two dimensions
        volume_data[:, :, idx] = slice_data

    # Save the volume as a NIfTI file
    affine = np.eye(4)  # Identity affine matrix
    nii_img = nib.Nifti1Image(volume_data, affine)
    nib.save(nii_img, output_nifti_path)
    print(f"NIfTI volume saved: {output_nifti_path}")


if __name__ == '__main__':
    # Specific patient ID for whom you want to reconstruct the NIfTI volume
    patient_id = '2BC090'

    # Directory containing TIFF slices for the test set
    tiff_dir = f'D:/test_results_pelvis_tiff/images'
    #tiff_dir = f'D:/test_results_pelvis_tiff'
    #tiff_dir = f'D:/test_inpainting_8bit/test_latest/images'

    # Output path for the reconstructed NIfTI volume
    output_nifti_path = f'D:/test_results_pelvis_tiff/{patient_id}.nii.gz'
    #output_nifti_path = f'D:/test_inpainting_8bit/test_latest/{patient_id}.nii.gz'

    # Convert TIFF slices to NIfTI volume
    tiff_to_nifti(patient_id, tiff_dir, output_nifti_path)


