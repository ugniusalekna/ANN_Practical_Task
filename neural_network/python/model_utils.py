import numpy as np
import os
import csv
import time
import torch
from scipy.ndimage import gaussian_filter, zoom
from skimage.filters import threshold_otsu
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from visualize_data import plot_numpy_matrices

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_unique_filename(base_path, base_name, ext):
    counter = 1
    while True:
        file_name = f"{base_name}_{counter}.{ext}"
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def monitor_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return allocated / (1024 ** 2), cached / (1024 ** 2)  # to megabytes
    else:
        return 0, 0

def create_meshgrid(height, width, device, num_sampled_points=None):

    y_coords, x_coords = torch.meshgrid(torch.linspace(0, height, 280), torch.linspace(0, width, 280), indexing='xy')
    coords = torch.stack([x_coords, y_coords], dim=0)

    if num_sampled_points == None:
        coords = coords.to(device)
        return coords.requires_grad_(True)

    flat_coords = coords.view(2, -1).transpose(0, 1)  # Shape: [280*280, 2]

    indices = torch.randperm(flat_coords.size(0))[:num_sampled_points]
    sampled_flat_coords = torch.zeros_like(flat_coords)
    sampled_flat_coords[indices] = flat_coords[indices]

    sampled_coords = sampled_flat_coords.transpose(0, 1).view(2, 280, 280).unsqueeze(0)
    sampled_coords = sampled_coords.to(device)

    return sampled_coords.requires_grad_(True)

def segment_vessel_otsu(velocity_field, pressure_field, gamma, sigma, target_size, device):
    def apply_blur(field, gamma, sigma):
        field_transformed = np.maximum(field, 0) ** gamma
        return gaussian_filter(field_transformed / np.max(field_transformed), sigma=sigma)

    def upscale(field, target_size):
        zoom_factors = [target_size[i] / field.shape[i+2] for i in range(2)]
        return zoom(field, zoom=[1, 1, *zoom_factors], order=3)  # Bicubic interpolation

    velocity_np = to_numpy(velocity_field)
    pressure_np = to_numpy(pressure_field)

    if velocity_np.shape[2:] != target_size:
        velocity_np = upscale(velocity_np, target_size)

    if pressure_np.shape[2:] != target_size:
        pressure_np = upscale(pressure_np, target_size)

    velocity_magnitude = np.linalg.norm(velocity_np, axis=1)
    blurred_velocity = apply_blur(velocity_magnitude, gamma, sigma)
    blurred_pressure = apply_blur(pressure_np.squeeze(1), gamma, sigma)

    velocity_mask = (blurred_velocity > threshold_otsu(blurred_velocity)).astype(np.float32)
    pressure_mask = (blurred_pressure > threshold_otsu(blurred_pressure)).astype(np.float32)

    combined_mask = np.maximum(velocity_mask[:, np.newaxis, :, :], pressure_mask[:, np.newaxis, :, :])

    return torch.from_numpy(combined_mask).float().to(device)

def compute_navier_stokes_loss(u_pred, p_pred, coordinates, mask, rho, mu, device):
    u, v, p = u_pred[:, [0]] * mask, u_pred[:, [1]] * mask, p_pred * mask

    du_dxy = torch.autograd.grad(u, coordinates, grad_outputs=torch.ones_like(u).to(device), retain_graph=True, create_graph=True)[0]
    dv_dxy = torch.autograd.grad(v, coordinates, grad_outputs=torch.ones_like(v).to(device), retain_graph=True, create_graph=True)[0]
    dp_dxy = torch.autograd.grad(p, coordinates, grad_outputs=torch.ones_like(p).to(device), retain_graph=True, create_graph=True)[0]

    d2u_dxy2 = torch.autograd.grad(du_dxy, coordinates, grad_outputs=torch.ones_like(du_dxy).to(device), create_graph=True)[0]
    d2v_dxy2 = torch.autograd.grad(dv_dxy, coordinates, grad_outputs=torch.ones_like(dv_dxy).to(device), create_graph=True)[0]

    du_dx, du_dy = du_dxy[:, [0]], du_dxy[:, [1]]
    dv_dx, dv_dy = dv_dxy[:, [0]], dv_dxy[:, [1]]
    dp_dx, dp_dy = dp_dxy[:, [0]], dp_dxy[:, [1]]
    d2u_dx2, d2u_dy2 = d2u_dxy2[:, [0]], d2u_dxy2[:, [1]]
    d2v_dx2, d2v_dy2 = d2v_dxy2[:, [0]], d2v_dxy2[:, [1]]

    continuity = du_dx + dv_dy
    continuity_loss = torch.max(torch.abs(continuity))

    momentum_u = rho * (u * du_dx + v * du_dy) + dp_dx - mu * (d2u_dx2 + d2u_dy2)
    momentum_v = rho * (u * dv_dx + v * dv_dy) + dp_dy - mu * (d2v_dx2 + d2v_dy2)
    momentum_loss = torch.max(torch.abs(momentum_u)) + torch.max(torch.abs(momentum_v))

    physics_loss = continuity_loss + momentum_loss

    return physics_loss / 1000

def train_model(model, train_loader, optimizer, criterion, num_epochs, device, alpha=0.5, model_save_path=None, loss_save_path=None, log_interval=10, is_physics_informed=False):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.train()

    all_data_losses = []
    all_physics_losses = []

    model_file_base_name = 'PICNN_model_params' if is_physics_informed else 'CNN_model_params'
    loss_file_base_name = 'PICNN_train_losses' if is_physics_informed else 'CNN_train_losses'
    
    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        save_model_file = get_unique_filename(model_save_path, model_file_base_name, 'pt')
        print(f'Model param file name: {save_model_file}')

    if loss_save_path is not None:
        os.makedirs(loss_save_path, exist_ok=True)
        save_loss_file = get_unique_filename(loss_save_path, loss_file_base_name, 'csv')
        print(f'Loss file name: {save_loss_file}')

    with open(save_loss_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Step', 'Data Loss', 'Physics Loss' if is_physics_informed else '', 'Loss'])

        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            running_loss = 0.0
            running_data_loss = 0.0
            running_physics_loss = 0.0
            
            for i, ((u_hr, p_hr), (u_lr, p_lr)) in enumerate(train_loader):
                u_hr, p_hr = u_hr.to(device), p_hr.to(device)
                u_lr, p_lr = u_lr.to(device), p_lr.to(device)

                optimizer.zero_grad()

                if is_physics_informed:
                    coordinates = create_meshgrid(0.006, 0.006, device, num_sampled_points=100).requires_grad_(True)
                    coordinates = coordinates.expand(u_lr.size(0), -1, -1, -1)
                    u_pred, p_pred = model(u_lr, p_lr, coordinates)
                    mask = segment_vessel_otsu(u_lr, p_lr, gamma=0.5, sigma=1.5, target_size=(280, 280), device=device)
                    data_loss = (1000 * criterion(u_pred, u_hr) + criterion(p_pred, p_hr))
                    physics_loss = compute_navier_stokes_loss(u_pred, p_pred, coordinates, mask, rho=1060, mu=0.0035, device=device)
                    loss = (1 - alpha) * data_loss + alpha * physics_loss
                else:
                    u_pred, p_pred = model(u_lr, p_lr)
                    data_loss = (1000 * criterion(u_pred, u_hr) + criterion(p_pred, p_hr))
                    loss = data_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_data_loss += data_loss.item() if is_physics_informed else loss.item()
                running_physics_loss += physics_loss.item() if is_physics_informed else 0

                if model_save_path is not None and (epoch * len(train_loader) + i) % 1000 == 0:
                    allocated_memory_MB, cached_memory_MB = monitor_gpu_memory()
                    print(f"Allocated Memory: {allocated_memory_MB:.2f} MB, Cached Memory: {cached_memory_MB:.2f} MB")
                    if i > 0:
                        print(f'Should be saving now!')
                        torch.save(model.state_dict(), save_model_file)
                        # u_lr_np, p_lr_np = to_numpy(u_lr), to_numpy(p_lr)
                        # u_hr_np, p_hr_np = to_numpy(u_hr), to_numpy(p_hr)
                        # u_pred_np, p_pred_np = to_numpy(u_pred), to_numpy(p_pred)

                        # for j in range(u_pred_np.shape[0]):
                        #     plot_numpy_matrices(u_lr_np[j].transpose(1, 2, 0), p_lr_np[j][0], main_title=f"Noisy: Epoch {epoch}, step {i}", plot_size=6)
                        #     plot_numpy_matrices(u_hr_np[j].transpose(1, 2, 0), p_hr_np[j][0], main_title=f"True: Epoch {epoch}, step {i}", plot_size=6)
                        #     plot_numpy_matrices(u_pred_np[j].transpose(1, 2, 0), p_pred_np[j][0], main_title=f"Predicted: Epoch {epoch}, step {i}", plot_size=6)


                if (i + 1) % log_interval == 0:
                    writer.writerow([epoch, i, data_loss.item(), physics_loss.item() if is_physics_informed else '', loss.item()])
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Total Loss: {running_loss / log_interval:.6f}, Data Loss: {running_data_loss / log_interval:.6f}, Physics Loss: {running_physics_loss / log_interval:.6f}')
                    all_data_losses.append(running_data_loss / log_interval)
                    if is_physics_informed:
                        all_physics_losses.append(running_physics_loss / log_interval)
                    
                    running_loss = 0.0
                    running_data_loss = 0.0
                    running_physics_loss = 0.0

            epoch_duration = time.time() - epoch_start_time
            print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} s')

    total_duration = time.time() - total_start_time
    print(f'Total training time: {total_duration:.2f} s')

    # Final save
    if model_save_path is not None:
        torch.save(model.state_dict(), save_model_file)
        print(f'Model params saved to: {save_model_file}')
        print(f'Losses saved to: {save_loss_file}')

    
    if is_physics_informed:
        return all_data_losses, all_physics_losses
    else:
        return all_data_losses

def test_model(model, test_loader, criterion, device, test_loss_save_path=None, log_interval=10, is_physics_informed=False):
    model.eval()
    total_loss = 0
    results = []
    
    if test_loss_save_path is not None:
        loss_file_base_name = 'PICNN_test_losses' if is_physics_informed else 'CNN_test_losses'
        os.makedirs(test_loss_save_path, exist_ok=True)
        save_loss_file = get_unique_filename(test_loss_save_path, loss_file_base_name, 'csv')

    with torch.no_grad(), open(save_loss_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Loss'])

        for i, ((u_hr, p_hr), (u_lr, p_lr)) in enumerate(test_loader):
            u_hr, p_hr = u_hr.to(device), p_hr.to(device)
            u_lr, p_lr = u_lr.to(device), p_lr.to(device)

            if is_physics_informed:
                coordinates = create_meshgrid(0.006, 0.006, device, num_sampled_points=100)
                coordinates = coordinates.expand(u_lr.size(0), -1, -1, -1)
                u_pred, p_pred = model(u_lr, p_lr, coordinates)
                loss = (100 * criterion(u_pred, u_hr) + criterion(p_pred, p_hr))
                
            else:
                u_pred, p_pred = model(u_lr, p_lr)
                loss = (100 * criterion(u_pred, u_hr) + criterion(p_pred, p_hr))

            total_loss += loss
            
            if (i + 1) % log_interval == 0:
                writer.writerow([i, loss.item()])
                print(f'Step [{i+1}/{len(test_loader)}], Total Loss: {loss.item():.6f}')

            results.append({
                'noisy': (u_lr.cpu().numpy(), p_lr.cpu().numpy()),
                'predicted': (u_pred.cpu().numpy(), p_pred.cpu().numpy()),
                'true': (u_hr.cpu().numpy(), p_hr.cpu().numpy())
            })

    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.6f}')

    return to_numpy(avg_loss), results

def evaluate_model(model, test_loader, device, is_physics_informed=False):
    model.eval()

    def psnr(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        max_pixel = 1.0  # Assuming pixel values range from 0 to 1
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    total_mse_velocity = total_psnr_velocity = total_ssim_velocity = 0
    total_mse_pressure = total_psnr_pressure = total_ssim_pressure = 0

    with torch.no_grad():
        for (u_hr, p_hr), (u_lr, p_lr) in test_loader:
            u_hr, p_hr = u_hr.to(device), p_hr.to(device)
            u_lr, p_lr = u_lr.to(device), p_lr.to(device)

            if is_physics_informed:
                coordinates = create_meshgrid(0.006, 0.006, device, num_sampled_points=100)
                coordinates = coordinates.expand(u_lr.size(0), -1, -1, -1)
                u_pred, p_pred = model(u_lr, p_lr, coordinates)
            else:
                u_pred, p_pred = model(u_lr, p_lr)
            
            u_hr_np, u_pred_np = to_numpy(u_hr), to_numpy(u_pred)
            p_hr_np, p_pred_np = to_numpy(p_hr), to_numpy(p_pred)
                                 
            for i in range(u_hr_np.shape[0]): # one batch
                
                u_hr_batch, u_pred_batch = u_hr_np[i], u_pred_np[i]
                for j in range(u_hr_batch.shape[0]):
                    total_mse_velocity += mean_squared_error(u_hr_batch[j], u_pred_batch[j])
                    total_psnr_velocity += psnr(u_hr_batch[j], u_pred_batch[j])
                    total_ssim_velocity += ssim(u_hr_batch[j], u_pred_batch[j], data_range=u_hr_batch[j].max()-u_hr_batch[j].min())                 

                p_hr_batch, p_pred_batch = p_hr_np[i].squeeze(0), p_pred_np[i].squeeze(0)
                total_mse_pressure += mean_squared_error(p_hr_batch, p_pred_batch)
                total_psnr_pressure += psnr(p_hr_batch, p_pred_batch)
                total_ssim_pressure += ssim(p_hr_batch, p_pred_batch, data_range=p_hr_batch.max()-p_hr_batch.min())
            

    avg_mse_velocity = total_mse_velocity / len(test_loader)
    avg_psnr_velocity = total_psnr_velocity / len(test_loader)
    avg_ssim_velocity = total_ssim_velocity / len(test_loader)

    avg_mse_pressure = total_mse_pressure / len(test_loader)
    avg_psnr_pressure = total_psnr_pressure / len(test_loader)
    avg_ssim_pressure = total_ssim_pressure / len(test_loader)

    velocity_metrics = [avg_mse_velocity, avg_psnr_velocity, avg_ssim_velocity]
    pressure_metrics = [avg_mse_pressure, avg_psnr_pressure, avg_ssim_pressure]
    
    return velocity_metrics, pressure_metrics
    
