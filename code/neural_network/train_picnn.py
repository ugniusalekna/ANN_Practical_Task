import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loading import FlowFieldDataset, train_test_split
from nn_model import SuperResolutionPICNN
from model_utils import train_model, test_model, evaluate_model
from visualize_data import plot_results, print_metrics, show_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name())
    torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


if __name__ == '__main__':

    npy_velocity_dir = 'data/blood_flow_simulations/velocity_fields/npy_data'
    # npy_velocity_dir = 'data/test/velocity_fields/npy_data'
    # npy_velocity_dir = 'data/test_one_case/velocity_fields/npy_data'

    dataset = FlowFieldDataset(npy_velocity_dir, load_all_data=False)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    batch_size = 1
    num_epochs = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # hidden_layers = np.array([16, 32, 32, 32, 32, 32, 32, 32, 16])
    hidden_layers = np.array([16, 32, 64, 128, 256, 128, 64, 32, 16])
    model = SuperResolutionPICNN(hidden_layers, enhance=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    alpha = 0.05
    is_physics_informed = True if alpha > 0.0 else False
    
    log_interval = 10
    model_save_path = 'logs/model_parameters'
    loss_save_path = 'logs/losses'
    test_loss_save_path = 'logs/test_losses'
    # model_load_path = 'results/CNN/run_1/CNN_model_params_47.pt'
    
    
    # - - - - - Training - - - - - 

    data_losses, physics_losses = train_model(model, train_loader, optimizer, criterion, num_epochs, device, alpha=alpha,
                                              model_save_path=model_save_path, loss_save_path=loss_save_path, log_interval=log_interval,
                                              is_physics_informed=is_physics_informed, norm='L2', plot_progress=False, add_constraints=False)

    # - - - - - Testing - - - - -
    
    avg_loss, results = test_model(model, test_loader, criterion, device, test_loss_save_path=test_loss_save_path, log_interval=log_interval,
                                   is_physics_informed=is_physics_informed)    
    velocity_metrics = evaluate_model(model, test_loader, device)

    # - - - - - Plotting - - - - -

    plot_results(num_epochs, results, data_losses, physics_losses, log_interval=log_interval, num_samples_to_plot=10)
    print_metrics(velocity_metrics)
    show_plots()