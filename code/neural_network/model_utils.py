import numpy as np
import os
import csv
import time
import torch
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from visualize_data import display_predicted_fields, plot_numpy_matrices, plot_with_transparent_mask
from image_processing import segment_vessel_otsu, gaussian_blur_tensor

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


def compute_derivative(tensor, dx, dy, order=1):
    derivative_x = torch.zeros_like(tensor)
    derivative_y = torch.zeros_like(tensor)

    # Central differences (interior)
    if order == 1:
        derivative_y[:, :, 1:-1, :] = (tensor[:, :, 2:, :] - tensor[:, :, :-2, :]) / (2 * dy)
        derivative_x[:, :, :, 1:-1] = (tensor[:, :, :, 2:] - tensor[:, :, :, :-2]) / (2 * dx)
    elif order == 2:
        derivative_y[:, :, 1:-1, :] = (tensor[:, :, 2:, :] - 2 * tensor[:, :, 1:-1, :] + tensor[:, :, :-2, :]) / (dy ** 2)
        derivative_x[:, :, :, 1:-1] = (tensor[:, :, :, 2:] - 2 * tensor[:, :, :, 1:-1] + tensor[:, :, :, :-2]) / (dx ** 2)

    # Forward differences (first row/col)
    if order == 1:
        derivative_y[:, :, 0, :] = (tensor[:, :, 1, :] - tensor[:, :, 0, :]) / dy
        derivative_x[:, :, :, 0] = (tensor[:, :, :, 1] - tensor[:, :, :, 0]) / dx
    elif order == 2:
        derivative_y[:, :, 0, :] = (tensor[:, :, 2, :] - 2 * tensor[:, :, 1, :] + tensor[:, :, 0, :]) / (dy ** 2)
        derivative_x[:, :, :, 0] = (tensor[:, :, :, 2] - 2 * tensor[:, :, :, 1] + tensor[:, :, :, 0]) / (dx ** 2)

    # Backward differences (last row/col)
    if order == 1:
        derivative_y[:, :, -1, :] = (tensor[:, :, -1, :] - tensor[:, :, -2, :]) / dy
        derivative_x[:, :, :, -1] = (tensor[:, :, :, -1] - tensor[:, :, :, -2]) / dx
    elif order == 2:
        derivative_y[:, :, -1, :] = (tensor[:, :, -1, :] - 2 * tensor[:, :, -2, :] + tensor[:, :, -3, :]) / (dy ** 2)
        derivative_x[:, :, :, -1] = (tensor[:, :, :, -1] - 2 * tensor[:, :, :, -2] + tensor[:, :, :, -3]) / (dx ** 2)

    return derivative_x, derivative_y

def compute_navier_stokes_loss(u_pred, mask, rho, mu, device, norm='L2'):

    u, v = u_pred[:, [0]], u_pred[:, [1]]

    # blur for smoother derivatives at boundaries
    kernel_size, sigma = 11, 5
    u, v = gaussian_blur_tensor(u, kernel_size, sigma, device), gaussian_blur_tensor(v, kernel_size, sigma, device)

    # Plot mask and masked regions
    # vector_field = torch.cat((u, v), dim=1)
    # for j in range(u.shape[0]):
    #     # plot_numpy_matrices(to_numpy(vector_field[j]), to_numpy(p[j][0]))
    #     plot_with_transparent_mask(to_numpy(vector_field[j]), to_numpy(mask[j][0]))

    dy, dx = 0.006 / u.shape[2], 0.006 / u.shape[3]
    # dy, dx = 0.006, 0.006
    
    du_dx, du_dy = compute_derivative(u, dx, dy, order=1)
    dv_dx, dv_dy = compute_derivative(v, dx, dy, order=1)

    omega = dv_dx - du_dy
    
    domega_dx, domega_dy = compute_derivative(omega, dx, dy, order=1)
    d2omega_dx2, d2omega_dy2 = compute_derivative(omega, dx, dy, order=1)

    du_dx *= mask
    du_dy *= mask
    dv_dx *= mask
    dv_dy *= mask
    domega_dx *= mask
    domega_dy *= mask
    d2omega_dx2 *= mask
    d2omega_dy2 *= mask

    continuity = du_dx + dv_dy
    momentum = rho * (u * domega_dx + v * domega_dy) - mu * (d2omega_dx2 + d2omega_dy2)

    match norm:
        case 'L1':
            continuity_loss = torch.sum(torch.abs(continuity))
            momentum_loss = torch.sum(torch.abs(momentum))
        case 'L2':
            continuity_loss = torch.sqrt(torch.sum(continuity**2))
            momentum_loss = torch.sqrt(torch.sum(momentum**2))
        case 'Linf':
            continuity_loss = torch.max(torch.abs(continuity))
            momentum_loss = torch.max(torch.abs(momentum))

    physics_loss = continuity_loss + momentum_loss
    
    return physics_loss

def apply_constraints(u_pred, mask, physics_loss):

    u, v = u_pred[:, [0]] * mask, u_pred[:, [1]] * mask

    params = {
        'lambda_ns':    1.2,
        'lambda_u':     0.2,
        'lambda_v':     0.1,
        'lambda_or':    5.0,
    }
    
    def variation_penalty(field):
        field_interior = field * mask
        std_dev_interior = torch.std(field_interior, dim=[2, 3])
        return -torch.mean(std_dev_interior)

    # 1. Penalize constant values inside the vessel
    u_penalty = variation_penalty(u)
    v_penalty = variation_penalty(v)

    # 2. Penalize non-zero values outside the vessel
    inverse_mask = 1 - mask
    or_penalty = torch.mean((u * inverse_mask) ** 2) + \
                 torch.mean((v * inverse_mask) ** 2)

    constrained_physics_loss = params['lambda_ns'] * physics_loss + \
                         params['lambda_u'] * u_penalty + params['lambda_v'] * v_penalty + params['lambda_or'] * or_penalty
    
    return constrained_physics_loss

def weighted_loss(u_pred, u_true, criterion, mask, beta):
    
    data_loss = criterion(u_pred, u_true)
    
    squared_diff = (u_pred - u_true) ** 2
    weighted_squared_diff = mask * squared_diff
    loss = weighted_squared_diff.mean()
    
    return beta * loss + (1 - beta) * data_loss

def compute_loss(model, u_hr, u_lr, criterion, device, alpha=0, is_physics_informed=False, norm='L2', add_constraints=False):
    
    data_scaling_factor = 100000
    
    u_pred = model(u_lr)
    mask = segment_vessel_otsu(u_hr, gamma=0.5, sigma=2.0, target_size=(280, 280), device=device)
    data_loss = data_scaling_factor * weighted_loss(u_pred, u_hr, criterion, mask, beta=0.5)
    # data_loss = criterion(u_pred, u_hr)

    if is_physics_informed and alpha > 0:
        physics_loss =  compute_navier_stokes_loss(u_pred, mask, rho=1060, mu=0.0035, device=device, norm=norm)
        # true_field_physics_loss = compute_navier_stokes_loss(u_hr, mask, rho=1060, mu=0.0035, device=device, norm=norm)
        loss = (1 - alpha) * data_loss + alpha * physics_loss
    else:
        loss = data_loss
        physics_loss = 0.0
        
        if add_constraints:
            physics_loss = apply_constraints(u_pred, mask, physics_loss)
        
    return [loss, data_loss, physics_loss], u_pred


def train_model(model, train_loader, optimizer, criterion, num_epochs, device, alpha=0,
                model_save_path=None, loss_save_path=None, log_interval=10, is_physics_informed=False, norm='L2',
                plot_progress=False, add_constraints=False):
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

            for i, (u_hr, u_lr) in enumerate(train_loader):
                u_hr, u_lr = u_hr.to(device), u_lr.to(device)

                optimizer.zero_grad()

                losses, u_pred = compute_loss(model, u_hr, u_lr, criterion, device, alpha, is_physics_informed, norm, add_constraints)
                loss, data_loss, physics_loss = losses
                
                loss.backward()
                optimizer.step()

                if model_save_path is not None and (epoch * len(train_loader) + i) % 1000 == 0:
                    allocated_memory_MB, cached_memory_MB = monitor_gpu_memory()
                    print(f"Allocated Memory: {allocated_memory_MB:.2f} MB, Cached Memory: {cached_memory_MB:.2f} MB")
                    if i > 0:
                        print(f'Model state saving at Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]')
                        torch.save(model.state_dict(), save_model_file)

                if (i + 1) % log_interval == 0:
                # if (epoch + 1) % log_interval == 0:

                    writer.writerow([epoch, i, data_loss.item(), physics_loss.item() if is_physics_informed and alpha > 0 else '', loss.item()])
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Total Loss: {loss:.6f}, Data Loss: {data_loss:.6f}, Physics Loss: {physics_loss:.6f}')
                    all_data_losses.append(data_loss.item())
                    if is_physics_informed:
                        all_physics_losses.append(physics_loss.item())

                    if plot_progress:
                        display_predicted_fields(u_lr, u_hr, u_pred, epoch, i)

            file.flush()
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

        for i, (u_hr, u_lr) in enumerate(test_loader):
            u_hr, u_lr = u_hr.to(device), u_lr.to(device)

            u_pred = model(u_lr)
            loss = 100 * criterion(u_pred, u_hr)

            total_loss += loss

            if (i + 1) % log_interval == 0:
                writer.writerow([i, loss.item()])
                print(f'Step [{i+1}/{len(test_loader)}], Data Loss: {loss.item():.6f}')

            results.append({
                'noisy': u_lr.cpu().numpy(),
                'predicted': u_pred.cpu().numpy(),
                'true': u_hr.cpu().numpy(),
            })
            
    if test_loss_save_path is not None:
        print(f'Test Losses saved to: {save_loss_file}')
    
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.6f}')

    return to_numpy(avg_loss), results

def evaluate_model(model, test_loader, device):
    model.eval()

    def psnr(y_true, y_pred):
        data_scaling_factor = 100000
        mse = data_scaling_factor * mean_squared_error(y_true, y_pred)
        return 20 * np.log10(np.max(y_true) / np.sqrt(mse))

    total_mse_velocity = total_psnr_velocity = total_ssim_velocity = 0

    with torch.no_grad():
        for (u_hr, u_lr) in test_loader:
            u_hr, u_lr = u_hr.to(device), u_lr.to(device)

            u_pred = model(u_lr)

            u_hr_np, u_pred_np = to_numpy(u_hr), to_numpy(u_pred)

            for i in range(u_hr_np.shape[0]): # one batch

                u_hr_batch, u_pred_batch = u_hr_np[i], u_pred_np[i]
                for j in range(u_hr_batch.shape[0]):
                    total_mse_velocity += mean_squared_error(u_hr_batch[j], u_pred_batch[j])
                    total_psnr_velocity += psnr(u_hr_batch[j], u_pred_batch[j])
                    total_ssim_velocity += ssim(u_hr_batch[j], u_pred_batch[j], data_range=u_hr_batch[j].max()-u_hr_batch[j].min())

    avg_mse_velocity = total_mse_velocity / len(test_loader)
    avg_psnr_velocity = total_psnr_velocity / len(test_loader)
    avg_ssim_velocity = total_ssim_velocity / len(test_loader)

    velocity_metrics = [avg_mse_velocity, avg_psnr_velocity, avg_ssim_velocity]

    return velocity_metrics