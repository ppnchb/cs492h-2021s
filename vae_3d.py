import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# class Conv(nn.Module):
#     def __init__(self, transpose, in_channels, out_channels, kernel_size, stride, padding, activation, normalize=True, skip=False, out_padding=0):
#         super(Conv, self).__init__()
        
#         assert not skip or (in_channels == out_channels)
        
#         if not transpose:
#             self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
#         else:
#             self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, out_padding)
#         self.activation = activation
#         self.normalization = None if not normalize else nn.BatchNorm3d(out_channels)
#         self.skip = skip
    
#     def forward(self, x0):
#         x1 = self.conv(x0)
#         x2 = x1 if self.normalization is None else self.normalization(x1)
#         x3 = x0 + x2 if self.skip else x2
#         x4 = self.activation(x3)
#         return x4

# class DoubleConvDown(nn.Module):
#     def __init__(self, in_channels, out_channels, activation):
#         super(DoubleConvDown, self).__init__()
#         self.conv0 = Conv(False, in_channels, in_channels, 3, 1, 1, activation, skip=True)
#         self.conv1 = Conv(False, in_channels, out_channels, 3, 2, 1, activation)
    
#     def forward(self, x0):
#         x1 = self.conv0(x0)
#         x2 = self.conv1(x1)
#         return x2

# class DoubleConvUp(nn.Module):
#     def __init__(self, in_channels, out_channels, activation):
#         super(DoubleConvUp, self).__init__()
#         self.conv0 = Conv(True, in_channels, out_channels, 3, 2, 1, activation, out_padding=1)
#         self.conv1 = Conv(True, out_channels, out_channels, 3, 1, 1, activation)
        
#     def forward(self, x0):
#         x1 = self.conv0(x0)
#         x2 = self.conv1(x1)
#         return x2
    
# class Encoder(nn.Module):
#     def __init__(self, channels=None):
#         super(Encoder, self).__init__()
        
#         self.num_layers = len(channels)
#         self.channels = channels
#         self.elu = nn.ELU()
        
#         layers = OrderedDict()
#         layers['init_conv'] = Conv(False, 1, channels[0], 3, 1, 1, self.elu)
#         for l in range(self.num_layers - 1):
#             in_channels = channels[l]
#             out_channels = channels[l+1]
#             layers[f'conv{l}'] = DoubleConvDown(in_channels, out_channels, self.elu)
#         layers['last_conv'] = Conv(False, channels[-1], 1, 1, 1, 0, self.elu)
#         layers['flatten'] = nn.Flatten()
#         self.model = nn.Sequential(layers)
    
#     def forward(self, x0):
#         return self.model(x0)

# class Decoder(nn.Module):
#     def __init__(self, initial_size, channels=None):
#         super(Decoder, self).__init__()
        
#         self.num_layers = len(channels)
#         self.channels = channels
#         self.elu = nn.ELU()
#         self.sigmoid = nn.Sigmoid()
        
#         layers = OrderedDict()
#         layers['unflatten'] = nn.Unflatten(1, (1, initial_size, initial_size, initial_size))
#         layers['init_conv'] = Conv(False, 1, channels[0], 1, 1, 0, self.elu)
#         for l in range(self.num_layers - 1):
#             in_channels = channels[l]
#             out_channels = channels[l+1]
#             layers[f'conv{l}'] = DoubleConvUp(in_channels, out_channels, self.elu)
#         layers['last_conv'] = Conv(False, channels[-1], 1, 1, 1, 0, self.sigmoid)
#         self.model = nn.Sequential(layers)
    
#     def forward(self, x0):
#         return self.model(x0)

# class VAE3D(nn.Module):
#     def __init__(self, input_size, latent_dims, channels=None):
#         super(VAE3D, self).__init__()
        
#         if channels is None:
#             channels = [8, 16, 32, 64, 128]
#         self.num_layers = len(channels)
#         self.channels = channels
#         self.input_size = input_size
#         self.output_size = input_size // (2 ** (self.num_layers - 1))
#         self.output_dims = self.output_size ** 3
#         self.latent_dims = latent_dims
        
#         self.encoder = Encoder(channels)
#         self.fc_mean = nn.Linear(self.output_dims, self.latent_dims)
#         self.fc_var = nn.Linear(self.output_dims, self.latent_dims)
#         self.fc_decode = nn.Linear(self.latent_dims, self.output_dims)
#         self.decoder = Decoder(self.output_size, channels[::-1])
    
#     def reparameterize(self, mean, logvar):
#         std_normal = torch.randn(*mean.shape, device=mean.device)
#         stdev = torch.exp(logvar * 0.5)
#         return mean + stdev * std_normal
    
#     def forward(self, x0):
#         x1 = self.encoder(x0)
        
#         mean = self.fc_mean(x1)
#         logvar = self.fc_var(x1)
#         z = self.reparameterize(mean, logvar)
        
#         x2 = self.fc_decode(z)
#         x3 = self.decoder(x2)
        
#         return x3, mean, logvar

class Conv(nn.Module):
    def __init__(self, transpose, in_channels, out_channels, kernel_size, stride, padding, activation, normalize=True, out_padding=0):
        super(Conv, self).__init__()
        
        if not transpose:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, out_padding)
        self.activation = activation
        self.normalization = None if not normalize else nn.BatchNorm3d(out_channels)
    
    def forward(self, x0):
        x1 = self.conv(x0)
        x2 = x1 if self.activation is None else self.activation(x1)
        x3 = x2 if self.normalization is None else self.normalization(x2)
        return x3

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation, normalize=True):
        super(Linear, self).__init__()
        self.activation = activation
        self.normalization = None if not normalize else nn.BatchNorm1d(out_channels)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x0):
        x1 = self.linear(x0)
        x2 = x1 if self.activation is None else self.activation(x1)
        x3 = x2 if self.normalization is None else self.normalization(x2)
        return x3

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super(Sine, self).__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        
        self.latent_dims = latent_dims
        
        self.elu = nn.ELU()
#         self.elu = nn.LeakyReLU()
        
        # Input: (B, 1, 32, 32, 32)
        self.conv1 = Conv(False, 1, 8, 3, 1, 0, self.elu) # (B, 8, 30, 30, 30)
        self.conv2 = Conv(False, 8, 16, 3, 2, 1, self.elu) # (B, 16, 15, 15, 15)
        self.conv3 = Conv(False, 16, 32, 3, 1, 0, self.elu) # (B, 32, 13, 13, 13)
        self.conv4 = Conv(False, 32, 64, 3, 2, 1, self.elu) # (B, 64, 7, 7, 7)
        self.flatten = nn.Flatten() # (B, 64*7*7*7)
        self.fc1 = Linear(64*7**3, 7**3, self.elu) # (B, 343)
        self.fc_mean = Linear(7**3, latent_dims, None) # (B, latent_dims)
        self.fc_logvar = Linear(7**3, latent_dims, None) # (B, latent_dims)
    
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.flatten(x4)
        x6 = self.fc1(x5)
        mean = self.fc_mean(x6)
        logvar = self.fc_logvar(x6)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        
        self.latent_dims = latent_dims
        
        self.elu = nn.ELU()
#         self.elu = nn.ReLU()
        
        # Input: (B, latent_dims)
        self.fc1 = Linear(latent_dims, 7**3, self.elu) # (B, 343)
        self.unflatten = nn.Unflatten(1, (1, 7, 7, 7)) # (B, 1, 7, 7, 7)
        self.conv1 = Conv(False, 1, 64, 3, 1, 1, self.elu) # (B, 64, 7, 7, 7)
        self.conv2 = Conv(True, 64, 32, 3, 2, 0, self.elu) # (B, 32, 15, 15, 15)
        self.conv3 = Conv(False, 32, 16, 3, 1, 1, self.elu) # (B, 16, 15, 15, 15)
        self.conv4 = Conv(True, 16, 8, 4, 2, 0, self.elu) # (B, 8, 32, 32, 32)
        self.conv5 = Conv(False, 8, 1, 3, 1, 1, None) # (B, 1, 32, 32, 32)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x0):
        x1 = self.fc1(x0)
        x2 = self.unflatten(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.sigmoid(x7)
        return x8

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.lrelu = nn.LeakyReLU()
        
        # Input: (B, 1, 32, 32, 32)
        self.conv1 = Conv(False, 1, 8, 3, 1, 0, self.lrelu, normalize=False) # (B, 8, 30, 30, 30)
        self.conv2 = Conv(False, 8, 16, 3, 2, 1, self.lrelu, normalize=False) # (B, 16, 15, 15, 15)
        self.conv3 = Conv(False, 16, 32, 3, 1, 0, self.lrelu, normalize=False) # (B, 32, 13, 13, 13)
        self.conv4 = Conv(False, 32, 64, 3, 2, 1, self.lrelu, normalize=False) # (B, 64, 7, 7, 7)
        self.flatten = nn.Flatten() # (B, 64*7*7*7)
        self.fc1 = Linear(64*7**3, 7**3, self.lrelu, normalize=False) # (B, 343)
        self.fc2 = Linear(7**3, 1, None, normalize=False)
    
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.flatten(x4)
        x6 = self.fc1(x5)
        x7 = self.fc2(x6)
        return x7
    
class VAE3D(nn.Module):
    def __init__(self, latent_dims):
        super(VAE3D, self).__init__()
        
        self.latent_dims = latent_dims
        
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)
    
    def reparameterize(self, mean, logvar):
        std_normal = torch.randn(*mean.shape, device=mean.device)
        stdev = torch.exp(0.5 * logvar)
        return mean + stdev * std_normal
    
    def forward(self, x0):
        mean, logvar = self.encoder(x0)
        
        z = self.reparameterize(mean, logvar)
        
        y = self.decoder(z)
        
        return y, mean, logvar

class VAE3DGAN(nn.Module):
    def __init__(self, latent_dims):
        super(VAE3DGAN, self).__init__()
        
        self.latent_dims = latent_dims
        
        self.autoencoder = VAE3D(latent_dims)
        self.discriminator = Discriminator()
    
    def forward(self, x0):
        y, mean, logvar = self.autoencoder(x0)
        
        dlogit_x0 = self.discriminator(x0)
        dlogit_y = self.discriminator(y)
        
        return y, mean, logvar, dlogit_x0, dlogit_y