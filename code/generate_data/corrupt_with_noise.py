import os
import numpy as np
import cv2


def add_gaussian_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def resize_data(data, target_resolution):
    if data.ndim == 2:
        return cv2.resize(data, (target_resolution[1], target_resolution[0]), interpolation=cv2.INTER_AREA)
    elif data.ndim == 3:
        channels = [cv2.resize(data[:, :, i], (target_resolution[1], target_resolution[0]), interpolation=cv2.INTER_AREA) for i in range(data.shape[2])]
        return np.stack(channels, axis=-1)
    
def process_and_save_files(base_dir, field_type, std, target_resolution):

    source_dir = os.path.join(base_dir, field_type, 'npy_data', 'true_data')
    target_dir = os.path.join(base_dir, field_type, 'npy_data', 'noisy_data')

    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(source_dir):
        if file_name.endswith('.npy'):

            data = np.load(os.path.join(source_dir, file_name))

            if len(data.shape) == 3:
                downsampled = resize_data(data, target_resolution)
                noisy_data = add_gaussian_noise(downsampled, std=std*np.max(data))

            else:
                downsampled = resize_data(data, target_resolution) 
                noisy_data = add_gaussian_noise(downsampled, std=std*np.max(data))
                
            np.save(os.path.join(target_dir, file_name), noisy_data)
            

base_dir = 'blood_flow_simulations'

process_and_save_files(base_dir, field_type='pressure_fields', std=0.065, target_resolution=(140, 140))
process_and_save_files(base_dir, field_type='velocity_fields', std=0.065, target_resolution=(140, 140))