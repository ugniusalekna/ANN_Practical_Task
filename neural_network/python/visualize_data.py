import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def plot_numpy_data(coordinates, connectivity, u_values, p_values, plot_size=8):

    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], connectivity)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    max_length = x_coords.max() - x_coords.min()
    max_height = y_coords.max() - y_coords.min()

    # plt.figure(figsize=(9, 2.25))
    plt.figure(figsize=(plot_size, int(np.round(plot_size*max_height/max_length))))
    plt.tripcolor(triangulation, np.sqrt(np.sum(u_values**2, axis=1)), shading='flat')
    plt.colorbar()
    plt.show()

    # plt.figure(figsize=(9, 2.25))
    plt.figure(figsize=(plot_size, int(np.round(plot_size*max_height/max_length))))
    plt.tripcolor(triangulation, p_values, shading='flat')
    plt.colorbar()
    plt.show()

def plot_numpy_matrices(u_values_np, p_values_np, main_title="Flow Field", plot_size=6):

    fig, axs = plt.subplots(1, 2, figsize=(plot_size * 2, plot_size))
    fig.suptitle(main_title, fontsize=16)

    magnitude = np.linalg.norm(u_values_np, axis=2)
    vel_img = axs[0].imshow(magnitude, cmap='viridis', norm=Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude)))    # origin='lower'
    axs[0].set_title('Velocity Magnitude')
    fig.colorbar(vel_img, ax=axs[0], fraction=0.046, pad=0.04)

    pres_img = axs[1].imshow(p_values_np, cmap='plasma', norm=Normalize(vmin=np.min(p_values_np), vmax=np.max(p_values_np)))
    axs[1].set_title('Pressure Field')
    fig.colorbar(pres_img, ax=axs[1], fraction=0.046, pad=0.04)

    domain_max = 0.006
    n_ticks = 5
    tick_locations = np.linspace(0, u_values_np.shape[0]-1, n_ticks)
    tick_labels = np.linspace(0, domain_max, n_ticks)

    for ax in axs:
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(['{:.3f}'.format(tl) for tl in tick_labels])
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(['{:.3f}'.format(tl) for tl in tick_labels])
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show(block=False)

    return fig, axs

def plot_with_transparent_mask(image, mask, alpha=0.5, plot_size=6):

  fig, axs = plt.subplots(1, 2, figsize=(plot_size * 2, plot_size))

  if len(image.shape) == 3:
    image = np.linalg.norm(image, axis=2)

  axs[0].imshow(image, cmap='gray', interpolation='none')
  axs[0].imshow(mask, cmap='jet', interpolation='none', alpha=alpha)
  axs[0].set_title('Transparent Mask On Top')

  axs[1].imshow(mask, cmap='viridis')
  axs[1].set_title('Mask')

  domain_max = 0.006
  n_ticks = 5
  tick_locations = np.linspace(0, image.shape[0]-1, n_ticks)
  tick_labels = np.linspace(0, domain_max, n_ticks)

  for ax in axs:
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(['{:.3f}'.format(tl) for tl in tick_labels])
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(['{:.3f}'.format(tl) for tl in tick_labels])
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])

  plt.show(block=False)

def create_gif_from_matched_files(velocity_dir, pressure_dir, matched_files, output_gif_path, fps=10):

  images = []
  processed_samples = 0

  for v_file, p_file in matched_files:
    print(processed_samples)
    if processed_samples >= 100:
      break
    u_values_np = np.load(os.path.join(velocity_dir, v_file))
    p_values_np = np.load(os.path.join(pressure_dir, p_file))

    fig, _ = plot_numpy_matrices(u_values_np, p_values_np, main_title=f'{v_file} and {p_file}')
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close(fig)

    processed_samples += 1

  imageio.mimsave(output_gif_path, images, fps=fps)
  
  
def plot_results(num_epochs, results, data_losses, physics_losses=None, log_interval=10, num_samples_to_plot=0):

    num_steps_per_epoch = len(data_losses) // num_epochs
    timesteps = [(epoch * num_steps_per_epoch + step) * log_interval 
                 for epoch in range(num_epochs) 
                 for step in range(num_steps_per_epoch)]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2 if physics_losses else 1, 1)
    plt.plot(timesteps, data_losses, label='Training Data Loss')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Data Loss')
    plt.legend()

    if physics_losses is not None:
        plt.subplot(1, 2, 2)
        plt.plot(timesteps, physics_losses, label='Training Physics Loss', color='orange')
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Physics Loss')
        plt.legend()

    plt.show(block=False)

    if num_samples_to_plot > 0:
      for i in range(min(num_samples_to_plot, len(results))):
          result = results[i]

          u_lr_np, p_lr_np = result['noisy']
          u_hr_np, p_hr_np = result['true']
          u_pred_np, p_pred_np = result['predicted']
          
          # u_lr_np, p_lr_np = to_numpy(u_lr), to_numpy(p_lr)
          # u_hr_np, p_hr_np = to_numpy(u_hr), to_numpy(p_hr)
          # u_pred_np, p_pred_np = to_numpy(u_pred), to_numpy(p_pred)

          for j in range(u_pred_np.shape[0]):
              plot_numpy_matrices(u_lr_np[j].transpose(1, 2, 0), p_lr_np[j][0], main_title="Noisy Flow Field", plot_size=6)
              plot_numpy_matrices(u_hr_np[j].transpose(1, 2, 0), p_hr_np[j][0], main_title="True Flow Field", plot_size=6)
              plot_numpy_matrices(u_pred_np[j].transpose(1, 2, 0), p_pred_np[j][0], main_title="Predicted Flow Field", plot_size=6)


def print_metrics(velocity_metrics, pressure_metrics):
  
    avg_mse_velocity, avg_psnr_velocity, avg_ssim_velocity = velocity_metrics
    avg_mse_pressure, avg_psnr_pressure, avg_ssim_pressure = pressure_metrics

    print("╔" + "═" * 60 + "╗")
    print("║ Results Summary                                            ║")                                                                                    
    print("╠" + "═" * 60 + "╣")
    print(f"║                   Average MSE: {avg_mse_velocity:.4f}")
    print(f"║     Velocity      Average PSNR: {avg_psnr_velocity:.4f} dB")
    print(f"║                   Average SSIM: {avg_ssim_velocity:.4f}")
    print("╠" + "═" * 60 + "╣")
    print(f"║                   Average MSE: {avg_mse_pressure:.4f}")
    print(f"║     Pressure      Average PSNR: {avg_psnr_pressure:.4f} dB")
    print(f"║                   Average SSIM: {avg_ssim_pressure:.4f}")
    print("╚" + "═" * 60 + "╝")
    
    
def show_plots():
  plt.show(block=True)
