import torch
import torch.nn as nn

class DeepONet(nn.Module): 
    def __init__(self, branch, trunk, output_dim: int = 1):  
        super().__init__() 
        
        self.branch = branch
        self.trunk  = trunk
        self.output_dim = output_dim

        if self.trunk.output_dim / self.branch.output_dim != self.output_dim:
            raise ValueError(f'The trunk net output dimension {self.trunk.output_dim} '
                             f'divided by the branch net output dimension {self.branch.output_dim} '
                             f'is not equal to the DeepONet output dimension {self.output_dim}.')

        self._last_biases = None
        self.init_params()

    @property
    def last_biases(self):
        if self._last_biases is None:
            self._last_biases = nn.Parameter(torch.zeros([self.output_dim]), requires_grad=True)
        return self._last_biases

    def init_params(self):
        self.branch.init_params()
        self.trunk.init_params()
        nn.init.zeros_(self.last_biases)
    
    def forward(self, branch_inputs, trunk_inputs):
        branch_outputs = self.branch(branch_inputs)
        trunk_outputs = self.trunk(trunk_inputs)   
        outputs = []
        for i in range(self.output_dim):
            output_component = torch.einsum('ik,jk->ij', branch_outputs, trunk_outputs[:, i*self.branch.output_dim:(i+1)*self.branch.output_dim]) 
            output_component = (output_component + self.last_biases[i])/ self.branch.output_dim**(1/2)
            outputs.append(output_component)
            
        outputs = torch.stack(outputs, dim=2)
        return outputs
