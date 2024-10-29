'''
This script prepares and splits the dataset of the MICCAI 2023 Grand Challenge SynthRad2023.
'''
import glob
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import imageio

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


def patient_nifti_to_png(nifti_path, patient_name, result_path):
    '''
    Turns the patient nifti file of an image modalities into numbered png files.
    '''

    # Load the nifti file
    nii_img = nib.load(nifti_path)
    nii_array = nii_img.get_fdata()

    # Normalizza i valori nell'intervallo 0-255
    min_val = -1024
    max_val = 3000
    nii_array_normalized = (nii_array - min_val) / (max_val - min_val) * 255
    nii_array_normalized = nii_array_normalized.astype('uint8')

    # Turn the nifti file into png files
    for idx, img_slice in enumerate(nii_array_normalized.transpose(2, 1, 0)):
        # Save the PNG file
        #transformed_slice = nonLinearLUT(img_slice)
        output_file = os.path.join(result_path, f'{patient_name}_{idx:03d}.png')
        imageio.imwrite(output_file, img_slice)

        print(f'PNG file saved: {output_file}')

def split_to_png(original_dataset_path, result_dataset_path, modality='ct'):
    '''
    Splits the dataset into train, validation and test sets and turns the nifti files into png files.
    '''

    # Split the dataset
    train_patients, val_patients, test_patients = split_patients(original_dataset_path, save_csv=False)

    # Create the folders for the png files
    train_path = os.path.join(result_dataset_path, 'train')
    val_path = os.path.join(result_dataset_path, 'val')
    test_path = os.path.join(result_dataset_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Turn the nifti files into png files
    for patient in train_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {train_patients.index(patient)+1}/{len(train_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, train_path)
    for patient in val_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {val_patients.index(patient)+1}/{len(val_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, val_path)
    for patient in test_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {test_patients.index(patient)+1}/{len(test_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, test_path)


def val_to_png(original_dataset_path, result_dataset_path, modality='ct'):
    '''
    Turns the nifti files of the validation set into png files without splitting the dataset in different folders
    '''

    # Create the folders for the png files
    if not os.path.exists(result_dataset_path):
        os.makedirs(result_dataset_path, exist_ok=True)


    # Turn the nifti files into png files
    val_patients = os.listdir(original_dataset_path)
    for patient in val_patients:
        # Prints the number of the patient and the total of patients being processed
        print(f'Patient {val_patients.index(patient)+1}/{len(val_patients)}')
        patient_path = os.path.join(original_dataset_path, patient)
        patient_nifti_to_png(os.path.join(patient_path, f'{modality}.nii.gz'), patient, result_dataset_path)


def pngs_to_nifti(image_path, result_path, original_nifti_path):
    original_nifti_image = nib.load(original_nifti_path)
    original_data = original_nifti_image.get_fdata()
    original_header = original_nifti_image.header

    # Get the list of .png files
    png_files_list = glob.glob(os.path.join(image_path, '*.png'))

    # Initialize an empty numpy array to hold the stacked data
    stacked_data = np.zeros(original_data.shape[:3] + (len(png_files_list),))

    # Load and stack the data from all the .png files
    for i, png_file in enumerate(png_files_list):
        png_image = Image.open(png_file)
        png_array = np.array(png_image)

        # Make sure the dimensions match between the .png and original NIfTI image
        if png_array.shape != original_data.shape[:2]:
            raise ValueError(f"The .png file {png_file} dimensions do not match the original NIfTI image.")

        stacked_data[..., i] = png_array

    # Create a new NIfTI image with the stacked data and original header
    new_nii_img = nib.Nifti1Image(stacked_data, original_data.affine, header=original_header)

    nib.save(new_nii_img, result_path)


def png_dataset_to_nifti(dataset_path, result_dataset_path, original_nifti_dataset_path):
    # for all of the patients in the dataset path turns their images into nifti files
    for patient in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient)
        result_patient_path = os.path.join(result_dataset_path, patient)
        os.makedirs(result_patient_path, exist_ok=True)
    return

if __name__ == '__main__':
    task = 'Task2'
    modality = 'cbct'
    structure = 'pelvis'
    if len(sys.argv) > 1:
        original_dataset_path = sys.argv[1]
        result_dataset_path = sys.argv[2]
    else:
        #original_dataset_path = f'D:/Datasets/{task}/{structure}'
        original_dataset_path = f'D:/Task2/{task}/{structure}'
        #result_dataset_path = f'D:/Datasets/{task}_png_uint8_fixed/{structure}_{modality}'
        result_dataset_path = f'D:/Task2/{task}_png_uint8/{structure}_{modality}'
    split_to_png(original_dataset_path, result_dataset_path, modality=modality)