import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
torch.manual_seed(0)

class BatchIndicesIterator:
    def __init__(self, start: int, end: int, batch_size: int, shuffle: bool=True):
        self.start = start
        self.end = end
        if  self.start >= self.end:
            raise ValueError(f'The start index {self.start} must be less than the end index {self.end}.')
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = torch.arange(start, end)
        self.num_indices = len(self.indices)
        self.current_index = 0

        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_indices)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.end - self.start:
            if self.shuffle:
                self.indices = self.indices[torch.randperm(self.num_indices)]
            self.current_index = 0
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_indices


class BaseDataset(Dataset):
    def __init__(self, data: numpy.ndarray | torch.Tensor):
        assert len(data.shape) >= 2, f'The input data must be at least a 2D array, got {len(data.shape)}D.'
        self.data = data
        if type(self.data) == numpy.ndarray:
            self.data = torch.from_numpy(self.data)
        self.data = self.data.to(torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

class BranchInputsDataset(BaseDataset):
    def __init__(self, branch_inputs: numpy.ndarray):
        super().__init__(branch_inputs)
        assert len(branch_inputs.shape) == 2, 'The branch inputs must be a 2D array.'

class TrunkInputsDataset(BaseDataset):
    def __init__(self, trunk_inputs: numpy.ndarray):
        super().__init__(trunk_inputs)
        assert len(trunk_inputs.shape) == 2, 'The trunk inputs must be a 2D array.'

class EvaluationLabelsDataset(BaseDataset):
    def __init__(self, evaluation_labels: numpy.ndarray):
        super().__init__(evaluation_labels)
        assert len(evaluation_labels.shape) == 3, 'The evaluation labels must be a 3D array.'

class DmLabelsDataset(BaseDataset):
    def __init__(self, dm_labels: numpy.ndarray):
        super().__init__(dm_labels)
        assert len(dm_labels.shape) == 4, 'The dm labels must be a 4D array.'

class DxLabelsDataset(BaseDataset):
    def __init__(self, dx_labels: numpy.ndarray):
        super().__init__(dx_labels)
        assert len(dx_labels.shape) == 4, 'The dx labels must be a 4D array.'

class CoeffLabelsDataset(BaseDataset):
    def __init__(self, coeff_labels: numpy.ndarray):
        super().__init__(coeff_labels)
        assert len(coeff_labels.shape) == 2, 'The coefficient labels must be a 2D array.'

class ReducedDmLabelsDataset(BaseDataset):
    def __init__(self, reduced_dm_labels: numpy.ndarray):
        super().__init__(reduced_dm_labels)
        assert len(reduced_dm_labels.shape) == 3, 'The reduced dm labels must be a 3D array.'


class BatchedDmFunctions:
    def __init__(self, model):
        self.model = model
        self.dm_function = torch.func.jacrev(model, argnums=0)
        self.batched_dm_functions = torch.vmap(self.dm_function, in_dims=(0, None), out_dims=0)

    def __call__(self, branch_inputs, trunk_inputs):
        is_training = self.model.training
        if is_training: # to avoid bug if the model has batch normalization [though we actually don't use batch normalization]
            self.model.eval()
        dm_outputs = self.batched_dm_functions(branch_inputs.unsqueeze(1), trunk_inputs)
        dm_outputs = dm_outputs.squeeze(1).squeeze(3)
        dm_outputs = dm_outputs.permute(0, 1, 3, 2)
        if is_training:
            self.model.train()
        return dm_outputs


class BatchedDxFunctions:
    def __init__(self, model):
        self.model = model
        self.dx_function = torch.func.jacrev(model, argnums=1)
        self.batched_dx_functions = torch.vmap(self.dx_function, in_dims=(None,0), out_dims=1)

    def __call__(self, branch_inputs, trunk_inputs):
        is_training = self.model.training
        if is_training: # to avoid bug if the model has batch normalization [though we actually don't use batch normalization]
            self.model.eval()
        dx_outputs = self.batched_dx_functions(branch_inputs, trunk_inputs.unsqueeze(1))
        dx_outputs = dx_outputs.squeeze(2).squeeze(3)
        dx_outputs = dx_outputs.permute(0, 1, 3, 2)
        if is_training:
            self.model.train()
        return dx_outputs


class BatchedReducedDmFunctions:
    def __init__(self, model):
        self.model = model
        self.batched_reduced_dm_functions = torch.vmap(torch.func.jacrev(model))

    def __call__(self, branch_inputs):
        is_training = self.model.training
        if is_training: # to avoid bug if the model has batch normalization [though we actually don't use batch normalization]
            self.model.eval()
        dm_outputs = self.batched_reduced_dm_functions(branch_inputs.unsqueeze(1))
        dm_outputs = dm_outputs.squeeze(1).squeeze(2)
        if is_training:
            self.model.train()
        return dm_outputs


class AddSpatialCoordinates(nn.Module):
    def __init__(self, num_x: int, num_y: int): 
        super().__init__()
        self.num_x, self.num_y = (num_x, num_y)
        self.x = numpy.linspace(0, 1, num_x)
        self.y = numpy.linspace(0, 1, num_y)
        self.x_coor, self.y_coor = numpy.meshgrid(self.x, self.y)
        self.x_coor = torch.from_numpy(self.x_coor).to(dtype=torch.float32)
        self.y_coor = torch.from_numpy(self.y_coor).to(dtype=torch.float32)

    def forward(self, inputs):
        repeat_x_coor = self.x_coor.reshape(1,1,self.num_x,self.num_y).repeat(inputs.shape[0],1,1,1)
        repeat_y_coor = self.y_coor.reshape(1,1,self.num_x,self.num_y).repeat(inputs.shape[0],1,1,1)

        return torch.cat((inputs, repeat_x_coor, repeat_y_coor), dim=1)