import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, zoom, binary_erosion, binary_opening

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def segment_vessel_otsu(velocity_field, gamma, sigma, target_size, device):

    def apply_blur(field, gamma, sigma):
        field_transformed = np.maximum(field, 0) ** gamma
        return gaussian_filter(field_transformed / np.max(field_transformed), sigma=sigma)
    def upscale(field, target_size):
        zoom_factors = [target_size[i] / field.shape[i+2] for i in range(2)]
        return zoom(field, zoom=[1, 1, *zoom_factors], order=3)  # Bicubic interpolation
    def apply_erosion(mask, erosion_size):
        structure = np.ones((erosion_size, erosion_size))
        return binary_erosion(mask, structure=structure)

    velocity_np = to_numpy(velocity_field)

    if velocity_np.shape[2:] != target_size:
        velocity_np = upscale(velocity_np, target_size)

    velocity_magnitude = np.linalg.norm(velocity_np, axis=1)
    blurred_velocity = apply_blur(velocity_magnitude, gamma, sigma)

    velocity_mask = (blurred_velocity > threshold_otsu(blurred_velocity)).astype(np.float32)

    velocity_mask = velocity_mask[:, np.newaxis, :, :]

    combined_mask_erosion = np.zeros_like(velocity_mask)
    for j in range(velocity_mask.shape[0]):
        combined_mask_erosion[j, 0] = apply_erosion(velocity_mask[j, 0], erosion_size=7)

    return torch.from_numpy(combined_mask_erosion).float().to(device)

# Gaussian blur for torch tensors
def gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = coords**2
    g = (-g / (2 * sigma**2)).exp()
    g /= g.sum()
    return g.view(1, -1) * g.view(-1, 1)

def gaussian_blur_tensor(tensor, kernel_size, sigma, device):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(device)
    kernel = kernel.expand(tensor.size(1), 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    blurred_tensor = F.conv2d(tensor, kernel, groups=tensor.size(1), padding=padding)
    return blurred_tensor