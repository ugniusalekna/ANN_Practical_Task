import os
import numpy as np

def augment_data(base_dir, subfolder, data_type, file_prefix, start_index=5015):
    
    data_dir = os.path.join(base_dir, subfolder, 'npy_data', data_type)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    for i, file in enumerate(files):
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)
        flipped_data = np.flip(data, axis=1)
        new_file_index = start_index + i
        new_file_name = f'{file_prefix}_{new_file_index}.npy'
        new_file_path = os.path.join(data_dir, new_file_name)
        np.save(new_file_path, flipped_data)
        print(f'Augumented file saved: {new_file_path}')


base_dir = 'data/blood_flow_simulations'

subfolders = ['velocity_fields', 'pressure_fields']
data_types = ['true_data', 'noisy_data']

augment_data(base_dir, subfolders[0], data_types[0], file_prefix='velocity')
augment_data(base_dir, subfolders[0], data_types[1], file_prefix='velocity')
augment_data(base_dir, subfolders[1], data_types[0], file_prefix='pressure')
augment_data(base_dir, subfolders[1], data_types[1], file_prefix='pressure')