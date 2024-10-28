import torch
import torch.nn as nn

from utils import get_activation, get_initializer

class FeedForwardNN(nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int,
                 hidden_depth: int,   
                 activation: str, 
                 init_func: str):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.activation_str = activation
        self.init_func_str = init_func

        self.activation = get_activation(self.activation_str)
        self.init_func = get_initializer(self.init_func_str)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_depth - 1)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.activation(self.input_layer(x))
        for _, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x
