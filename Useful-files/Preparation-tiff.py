import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
import tifffile as tiff


def split_patients(original_dataset_path, train_size=0.80, val_size=0.10, save_csv=True):
    '''
    Splits the dataset into train, validation and test sets.
    '''
    patients = os.listdir(original_dataset_path)
    patients.sort()
    np.random.seed(42)
    np.random.shuffle(patients)
    num_patients = len(patients)

    train_split = int(np.floor(train_size * num_patients))
    val_split = int(np.floor((train_size + val_size) * num_patients))
    train_patients = patients[:train_split]
    val_patients = patients[train_split:val_split]
    test_patients = patients[val_split:]

    # Save the splits in a csv file
    if save_csv:
        df = pd.DataFrame({'train': train_patients, 'val': val_patients, 'test': test_patients})
        df.to_csv(os.path.join(original_dataset_path, 'split.csv'), index=False)
    return train_patients, val_patients, test_patients


def patient_nifti_to_tiff(nifti_path, patient_name, result_path):
    '''
    Turns the patient nifti file of an image modalities into numbered tiff files.
    '''

    # Load the nifti file
    nii_img = nib.load(nifti_path)
    nii_array = nii_img.get_fdata()
    # Normalizza i valori nell'intervallo 0-1
    min_val = -1024
    max_val = 3000
    nii_array_normalized = (nii_array - min_val) / (max_val - min_val)
    # Turn the nifti file into tiff files
    for idx, img_slice in enumerate(nii_array_normalized.transpose(2, 1, 0)):
        img_slice_32bit = img_slice.astype(np.float32)
        # Save the slice as TIFF
        output_file = os.path.join(result_path, f'{patient_name}_{idx:03d}.tif')
        tiff.imwrite(output_file, img_slice_32bit,  dtype='float32', )

def split_to_tiff(original_dataset_path, result_dataset_path, modality='ct'):
    '''
    Splits the dataset into train, validation and test sets and turns the nifti files into tiff files.
    '''

    # Split the dataset
    train_patients, val_patients, test_patients = split_patients(original_dataset_path, save_csv=False)

    # Create the folders for the tiff files
    train_path = os.path.join(result_dataset_path, 'train')
    val_path = os.path.join(result_dataset_path, 'val')
    test_path = os.path.join(result_dataset_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Turn the nifti files into tiff files
    for patient in train_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {train_patients.index(patient)+1}/{len(train_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_tiff(os.path.join(patient_path, f'{modality}.nii.gz'), patient, train_path)
    for patient in val_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {val_patients.index(patient)+1}/{len(val_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_tiff(os.path.join(patient_path, f'{modality}.nii.gz'), patient, val_path)
    for patient in test_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {test_patients.index(patient)+1}/{len(test_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_tiff(os.path.join(patient_path, f'{modality}.nii.gz'), patient, test_path)


if __name__ == '__main__':
    task = 'Task2'
    modality = 'ct'
    structure = 'pelvis'
    if len(sys.argv) > 1:
        original_dataset_path = sys.argv[1]
        result_dataset_path = sys.argv[2]
    else:
        original_dataset_path = f'D:/Task2/{task}/{structure}'
        result_dataset_path = f'D:/Task2/{task}_tiff_float32/{structure}_{modality}'
    split_to_tiff(original_dataset_path, result_dataset_path, modality=modality)