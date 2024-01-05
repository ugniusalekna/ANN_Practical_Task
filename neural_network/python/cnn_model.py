import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loading import FlowFieldDataset, train_test_split
from nn_modules import SuperResolutionCNN
from model_utils import train_model, test_model, evaluate_model
from visualize_data import plot_results, print_metrics, show_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name())
    torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == '__main__':
       

    npy_velocity_dir = 'blood_flow_simulations/velocity_fields/npy_data'
    npy_pressure_dir = 'blood_flow_simulations/pressure_fields/npy_data'

    # npy_velocity_dir = 'test/velocity_fields/npy_data'
    # npy_pressure_dir = 'test/pressure_fields/npy_data'

    dataset = FlowFieldDataset(npy_velocity_dir, npy_pressure_dir, load_all_data=False)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    batch_size = 4
    num_epochs = 100

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    hidden_layers = np.array([16, 32, 32, 64, 32, 32, 16]) # 32, 32, 32, 32, 32, 32, 32, 
    model = SuperResolutionCNN(hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    model_save_path = 'model_parameters'
    loss_save_path = 'losses'
    log_interval = 10
    
    # - - - - - Training - - - - - 

    data_losses = train_model(model, train_loader, optimizer, criterion, num_epochs, device,
                              model_save_path=model_save_path, loss_save_path=loss_save_path, log_interval=log_interval)

    # - - - - - Testing - - - - -

    # model.load_state_dict(torch.load('neural_network\parameters_backup\CNN_model_params_1st_run.pt'))
    avg_loss, results = test_model(model, test_loader, criterion, device, test_loss_save_path='losses', log_interval=log_interval)
    velocity_metrics, pressure_metrics = evaluate_model(model, test_loader, device)

    # - - - - - Plotting - - - - -
    
    plot_results(num_epochs, results, data_losses, log_interval=log_interval, num_samples_to_plot=0)
    print_metrics(velocity_metrics, pressure_metrics)
    show_plots()