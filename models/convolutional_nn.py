import torch
import torch.nn as nn

from utils import get_activation, get_initializer

# Desinged for inputs with shape (batch_size, 1, height, width)
class ConvolutionalNN_65(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (65,65)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=3),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            self.activation,
            nn.Flatten(),
            nn.Linear(512*2*2, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, *self.input_dim)
        x = self.layers(x)
        return x

# Desinged for inputs with shape (batch_size, 1, height, width)
class ConvolutionalNN_49(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (49,49)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            self.activation,
            nn.Flatten(),
            nn.Linear(512, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, *self.input_dim)
        x = self.layers(x)
        return x



# Desinged for inputs with shape (batch_size, 1, height, width)
class ConvolutionalNN_33(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (33,33)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            self.activation,
            nn.Flatten(),
            nn.Linear(512, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, *self.input_dim)
        x = self.layers(x)
        return x


# Desinged for inputs with shape (batch_size, 1, height, width)
class ConvolutionalNN_81(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (81,81)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            self.activation,
            nn.Flatten(),
            nn.Linear(512*2*2, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, *self.input_dim)
        x = self.layers(x)
        return x


class ConvolutionalNN_101(nn.Module):

    def __init__(self, 
                 output_dim,
                 activation: str,
                 init_func: str):

        super().__init__()
        
        self.input_dim = (101,101)
        self.output_dim = output_dim
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=5, stride=3),
            self.activation,
            nn.Flatten(),
            nn.Linear(512*2*2, 512),
            self.activation,
            nn.Linear(512, output_dim)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, *self.input_dim)
        x = self.layers(x)
        return x
