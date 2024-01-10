import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def gaussian_kernel(size=3, sigma=1):
    gauss = np.exp(-np.linspace(-(size // 2), size // 2, size) ** 2 / (2 * sigma ** 2))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()

    return kernel

class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size=3, sigma=1):
        super(GaussianBlur, self).__init__()
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False) # Change to false
        self.groups = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=self.groups, bias=False, padding=kernel_size//2)

    def forward(self, x):
        self.conv.weight = self.weight
        return self.conv(x)

class FrequencyFilter(nn.Module):
    def __init__(self, threshold):
        super(FrequencyFilter, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        x = x.float()
        batch_size, channels, _, _ = x.shape
        denoised = torch.zeros_like(x)

        for c in range(channels):
            for b in range(batch_size):

                f = fft.fftn(x[b, c])
                fshift = fft.fftshift(f)

                magnitude = torch.abs(fshift)
                phase = torch.angle(fshift)

                mask = magnitude > self.threshold
                magnitude_filtered = magnitude * mask
                fshift_filtered = magnitude_filtered * torch.exp(1j * phase)

                f_ishift = fft.ifftshift(fshift_filtered)
                img_back = fft.ifftn(f_ishift)
                denoised[b, c] = torch.abs(img_back)

        return denoised


class SuperResolutionPICNN(nn.Module):
    def __init__(self, hidden_layers, enhance=False, dropout_rate=0.2):
        super(SuperResolutionPICNN, self).__init__()

        # self.activation = Swish()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.input_bn = nn.BatchNorm2d(hidden_layers[0])
        self.blur = GaussianBlur(channels=2, kernel_size=3, sigma=0.20)       # sigma=.25 for coarse data
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.denoise = FrequencyFilter(threshold=1.75)                         # threshold=.3 for coarse data
        self.enhance = enhance

        self.input_conv = nn.Conv2d(2, hidden_layers[0], kernel_size=3, padding=1)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.conv_layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(hidden_layers[i]))

        self.output_conv = nn.Conv2d(hidden_layers[-1], 2, kernel_size=3, padding=1)

    def forward(self, velocity):
        
        if self.enhance:
            velocity = self.denoise(velocity)
            velocity = self.blur(velocity)
        
        velocity = self.upsample(velocity)

        velocity = self.input_conv(velocity)
        velocity = self.activation(self.input_bn(velocity))
        velocity = self.dropout(velocity)

        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            velocity = conv_layer(velocity)
            velocity = self.activation(bn_layer(velocity))
            velocity = self.dropout(velocity)
        velocity_output = self.output_conv(velocity)

        return velocity_output