import numpy as np
import torch
import torch.nn as nn

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
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.groups = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=self.groups, bias=False, padding=kernel_size//2)

    def forward(self, x):
        self.conv.weight = self.weight
        return self.conv(x)

class SuperResolutionCNN(nn.Module):
    def __init__(self, hidden_layers, dropout_rate=0.2, blur_kernel_size=5, blur_sigma=1.5):
        super(SuperResolutionCNN, self).__init__()
        
        # self.activation = Swish()
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm2d(hidden_layers[0])

        self.velocity_blur = GaussianBlur(channels=2, kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.pressure_blur = GaussianBlur(channels=1, kernel_size=blur_kernel_size, sigma=blur_sigma)

        self.velocity_input_conv = nn.Conv2d(2, hidden_layers[0], kernel_size=3, padding=1)
        self.pressure_input_conv = nn.Conv2d(1, hidden_layers[0], kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=7, mode='bicubic', align_corners=True)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(1, len(hidden_layers)):
            self.conv_layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(hidden_layers[i]))

        self.velocity_output_conv = nn.Conv2d(hidden_layers[-1], 2, kernel_size=3, padding=1)
        self.pressure_output_conv = nn.Conv2d(hidden_layers[-1], 1, kernel_size=3, padding=1)

    def forward(self, velocity_input, pressure_input):

        velocity = self.velocity_blur(velocity_input)
        velocity = self.upsample(velocity)
        # velocity = self.velocity_blur(velocity)
        velocity = self.velocity_input_conv(velocity)
        velocity = self.activation(self.batch_norm(velocity))
        velocity = self.dropout(velocity)
        # velocity = self.upsample(velocity)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            velocity = conv_layer(velocity)
            velocity = self.activation(bn_layer(velocity))
            velocity = self.dropout(velocity)
        velocity_output = self.velocity_output_conv(velocity)

        pressure = self.pressure_blur(pressure_input)
        pressure = self.upsample(pressure)
        # pressure = self.pressure_blur(pressure)
        pressure = self.pressure_input_conv(pressure)
        pressure = self.activation(self.batch_norm(pressure))
        pressure = self.dropout(pressure)
        # pressure = self.upsample(pressure)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            pressure = conv_layer(pressure)
            pressure = self.activation(bn_layer(pressure))
            pressure = self.dropout(pressure)
        pressure_output = self.pressure_output_conv(pressure)

        return velocity_output, pressure_output

class SuperResolutionPICNN(nn.Module):
    def __init__(self, hidden_layers, dropout_rate=0.2, blur_kernel_size=5, blur_sigma=1.5):
        super(SuperResolutionPICNN, self).__init__()

        # self.activation = Swish()
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.input_bn = nn.BatchNorm2d(hidden_layers[0])

        self.velocity_blur = GaussianBlur(channels=2, kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.pressure_blur = GaussianBlur(channels=1, kernel_size=blur_kernel_size, sigma=blur_sigma)

        self.velocity_input_conv = nn.Conv2d(2 + 2, hidden_layers[0], kernel_size=3, padding=1)  # 2 velocity channels + 2 coordinates channel
        self.pressure_input_conv = nn.Conv2d(1 + 2, hidden_layers[0], kernel_size=3, padding=1)  # 1 pressure channel + 2 coordinates channel

        self.coords_conv = nn.Conv2d(2, hidden_layers[0], kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=7, mode='bicubic', align_corners=True)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(1, len(hidden_layers)):
            self.conv_layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(hidden_layers[i]))

        self.velocity_output_conv = nn.Conv2d(hidden_layers[-1], 2, kernel_size=3, padding=1)
        self.pressure_output_conv = nn.Conv2d(hidden_layers[-1], 1, kernel_size=3, padding=1)

    def forward(self, velocity_input, pressure_input, coords):
        velocity = self.velocity_blur(velocity_input)
        velocity = self.upsample(velocity)
        velocity = self.velocity_blur(velocity)
        velocity = torch.cat([velocity, coords], dim=1)
        velocity = self.velocity_input_conv(velocity)
        velocity = self.activation(self.input_bn(velocity))
        velocity = self.dropout(velocity)

        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            velocity = conv_layer(velocity)
            velocity = self.activation(bn_layer(velocity))
            velocity = self.dropout(velocity)
        velocity_output = self.velocity_output_conv(velocity)

        pressure = self.pressure_blur(pressure_input)
        pressure = self.upsample(pressure)
        pressure = self.pressure_blur(pressure)
        pressure = torch.cat([pressure, coords], dim=1)
        pressure = self.pressure_input_conv(pressure)
        pressure = self.activation(self.input_bn(pressure))
        pressure = self.dropout(pressure)

        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            pressure = conv_layer(pressure)
            pressure = self.activation(bn_layer(pressure))
            pressure = self.dropout(pressure)
        pressure_output = self.pressure_output_conv(pressure)

        return velocity_output, pressure_output