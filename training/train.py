import time
import math
import copy

import numpy
import torch
from utils import format_elapsed_time

from .training_utils import BatchedDmFunctions, BatchedDxFunctions, BatchedReducedDmFunctions
from .training_utils import BatchIndicesIterator, EvaluationLabelsDataset, DmLabelsDataset, DxLabelsDataset, CoeffLabelsDataset, ReducedDmLabelsDataset # noqa

# trainer for de_deeponet
class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_functions, loss_balancing_algorithm=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_functions = loss_functions
        self.loss_balancing_algorithm = loss_balancing_algorithm

        self._batched_dm_functions = None
        self._batched_dx_functions = None

    @property
    def batched_dm_functions(self):
        if self._batched_dm_functions is None:
            self._batched_dm_functions = BatchedDmFunctions(self.model)
        return self._batched_dm_functions

    @property
    def batched_dx_functions(self):
        if self._batched_dx_functions is None:
            self._batched_dx_functions = BatchedDxFunctions(self.model)
        return self._batched_dx_functions


    def _train_or_evaluate(self, mode, inputs, labels, batch_size, sampling_fraction, loss_weights, device, debug=True, shuffle=True, return_outputs=False, disable_lr_scheduler=False):
        """
        Args:
        mode = 'train' | 'evaluate'
        inputs = {
            'branch': BranchInputsDataset(branch_inputs),
            'trunk': TrunkInputsDataset(trunk_inputs),
        }
        labels = {
            'eval': EvaluationLabelsDataset(eval_labels) | None,
            'dm': DmLabelsDataset(dm_labels) | None,
            'dx': DxLabelsDataset(dx_labels) | None,
        }
        batch_size = {
            'branch': 8 | 'all',
            'trunk': 1024 | 'all'
        }
        sampling_fraction = 0.15
        loss_weights = {
            'eval': 1.0 | None,
            'dm': 1.0 | None,
            'dx': 1.0 | None
        }
        device = torch.device('cuda:3')
        debug = True
        """
        if mode not in ['train', 'evaluate']:
            raise ValueError(f"Invalid mode: {mode}. Mode must be either 'train' or 'evaluate'.")

        avg_losses = {key: 0.0 for key in ['eval', 'dm', 'dx', 'total']}
        num_functions = len(inputs['branch'])
        if batch_size['branch'] == 'all' or batch_size['branch'] > num_functions:
            batch_size['branch'] = num_functions
            if debug:
                print(f'The batch size for branch net is set to {num_functions}')

        num_evaluation_points = len(inputs['trunk'])
        if batch_size['trunk'] == 'all' or batch_size['trunk'] > num_evaluation_points:
            batch_size['trunk'] = num_evaluation_points


        if sampling_fraction > 1.0 or sampling_fraction < 0.0:
            raise ValueError(f'sampling_fraction must be in [0.0, 1.0], but got {sampling_fraction}')

        num_evaluation_points_each_iteration = math.ceil(sampling_fraction * num_evaluation_points)

        branch_indices_iterator = BatchIndicesIterator(0, len(inputs['branch']), batch_size['branch'], shuffle=shuffle)
        trunk_indices_iterator = BatchIndicesIterator(0, len(inputs['trunk']), batch_size['trunk'], shuffle=shuffle)

        if return_outputs:
            outputs = {key: [] for key in ['eval', 'dm', 'dx']}

        self.model = self.model.to(device)
        self.model.train() if mode == 'train' else self.model.eval()
        start_time = time.time()
        for batched_branch_indices in branch_indices_iterator:
            counter = 0 
            for batched_trunk_indices in trunk_indices_iterator:
                if counter >= num_evaluation_points_each_iteration:
                    break
                else:
                    if len(batched_trunk_indices) > num_evaluation_points_each_iteration - counter:
                        batched_trunk_indices = batched_trunk_indices[: int(num_evaluation_points_each_iteration - counter)]

                    counter += len(batched_trunk_indices)

                    batched_inputs = {
                        'branch': inputs['branch'][batched_branch_indices, :].to(device),
                        'trunk': inputs['trunk'][batched_trunk_indices, :].to(device)
                    }
                    keys = ['eval', 'dm', 'dx']
                    batched_labels = {key: None for key in keys}
                    batched_outputs = {key: None for key in keys}
                    losses = {key: None for key in keys}
                    for key in keys:
                        if labels[key] is not None and loss_weights[key] is not None and self.loss_functions[key] is not None:
                            if key == 'eval':
                                batched_labels[key] = labels[key][batched_branch_indices, :, :][:, batched_trunk_indices, :].to(device)
                                batched_outputs[key] = self.model(batched_inputs['branch'], batched_inputs['trunk'])
                            elif key == 'dm':
                                batched_labels[key] = labels[key][batched_branch_indices, :, :, :][:, batched_trunk_indices, :, :].to(device)
                                batched_outputs[key] = self.batched_dm_functions(batched_inputs['branch'], batched_inputs['trunk'])
                            elif key == 'dx':
                                batched_labels[key] = labels[key][batched_branch_indices, :, :, :][:, batched_trunk_indices, :, :].to(device)
                                batched_outputs[key] = self.batched_dx_functions(batched_inputs['branch'], batched_inputs['trunk'])
                            losses[key] = self.loss_functions[key](batched_outputs[key], batched_labels[key])
                            avg_losses[key] += loss_weights[key] * losses[key].item()
                        else:
                            avg_losses[key] = None

                    if all(value is None for value in losses.values()):
                        raise ValueError("All values in 'losses' are None. Please check the arguments 'labels', 'loss_functions' and 'loss_weights'.")

                    if mode == 'train':
                        if self.loss_balancing_algorithm is not None:
                            self.loss_balancing_algorithm.rebalance(losses, loss_weights)

                    total_loss = sum(loss_weights[key] * value for key, value in losses.items() if value is not None)
                    avg_losses['total'] += total_loss.item()

                    if mode == 'train':
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()

                    if return_outputs:
                        for key, value in batched_outputs.items():
                            if value is not None:
                                outputs[key].append(value)
                            else:
                                outputs[key] = None

        if mode == 'train' and not disable_lr_scheduler:
            self.lr_scheduler.step()

        for key, value in avg_losses.items():
            if value is not None:
                avg_losses[key] /= num_functions

        if return_outputs:
            num_batches = {
                'branch': math.ceil(num_functions / batch_size['branch']),
                'trunk': math.ceil(num_evaluation_points / batch_size['trunk'])
            }
            temp = {key: [] for key in ['eval', 'dm', 'dx']}
            for key, value in outputs.items():
                if value is not None:
                    for i in range(num_batches['branch']):
                        temp[key].append(torch.cat(value[i * num_batches['trunk']: (i+1) * num_batches['trunk']], dim=1))
                else:
                    temp[key] = None
            for key, value in temp.items():
                if value is not None:
                    outputs[key] = torch.cat(value, dim=0)
                    if key == 'eval':
                        outputs[key] = EvaluationLabelsDataset(outputs[key])
                    elif key == 'dm':
                        outputs[key] = DmLabelsDataset(outputs[key])
                    elif key == 'dx':
                        outputs[key] = DxLabelsDataset(outputs[key])
                else:
                    outputs[key] = None

        end_time = time.time()
        if debug:
            print(f'Time [{mode}]: {format_elapsed_time(start_time, end_time)}')
            print(f'Losses [{mode}]: {avg_losses}')
            print(f'Loss weights: {loss_weights}')
            print('')

        if return_outputs:
            return avg_losses, outputs 
        return avg_losses


    def train(self, inputs, labels, batch_size, sampling_fraction, loss_weights, device, debug=True, shuffle=True, return_outputs=False, disable_lr_scheduler=False):
        return self._train_or_evaluate('train', inputs, labels, batch_size, sampling_fraction, loss_weights, device, debug, shuffle, return_outputs, disable_lr_scheduler)

    def evaluate(self, inputs, labels, batch_size, sampling_fraction, loss_weights, device, debug=True, shuffle=False, return_outputs=False, disable_lr_scheduler=False):
        with torch.no_grad():
            return self._train_or_evaluate('evaluate', inputs, labels, batch_size, sampling_fraction, loss_weights, device, debug, shuffle, return_outputs, disable_lr_scheduler)

    def post_process(self, outputs, labels, labels_stats):
        """
        Args:
        outputs = {
            'eval': EvaluationLabelsDataset(eval_outputs) | None,
            'dm': DmLabelsDataset(dm_outputs) | None,
            'dx': DxLabelsDataset(dx_outputs) | None,
        }
        labels = {
            'eval': EvaluationLabelsDataset(eval_labels) | None,
            'dm': DmLabelsDataset(dm_labels) | None,
            'dx': DxLabelsDataset(dx_labels) | None,
        }
        labels_stats = {
            'mean': float,
            'std': float
        }
        Returns:
        outputs = {
            'eval': numpy.ndarray | None,
            'dm': numpy.ndarray | None,
            'dx': numpy.ndarray | None,
        }
        labels = {
            'eval': numpy.ndarray | None,
            'dm': numpy.ndarray | None,
            'dx': numpy.ndarray | None,
        }
        """
        def helper(data: EvaluationLabelsDataset | DmLabelsDataset | DxLabelsDataset | None, add_mean: bool=False)->numpy.ndarray | None:
            if data is not None:
                data = data[:].detach().cpu().numpy() if data[:].device.type == 'cuda' else data[:].numpy()
                data = data * labels_stats['std']
                if add_mean:
                    data += labels_stats['mean']
            return data
        outputs['eval'] = helper(outputs['eval'], add_mean=True)
        outputs['dm'] = helper(outputs['dm'])
        outputs['dx'] = helper(outputs['dx'])

        labels['eval'] = helper(labels['eval'], add_mean=True)
        labels['dm'] = helper(labels['dm'])
        labels['dx'] = helper(labels['dx'])
        return outputs, labels



# trainer for dino
class Trainer_v2:
    def __init__(self, model, optimizer, lr_scheduler, loss_functions, loss_balancing_algorithm=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_functions = loss_functions
        self.loss_balancing_algorithm = loss_balancing_algorithm

        self._batched_reduced_dm_functions = None

    @property
    def batched_reduced_dm_functions(self):
        if self._batched_reduced_dm_functions is None:
            self._batched_reduced_dm_functions = BatchedReducedDmFunctions(self.model)
        return self._batched_reduced_dm_functions


    def _train_or_evaluate(self, mode, inputs, labels, batch_size, loss_weights, device, debug=True, shuffle=True, return_outputs=False, disable_lr_scheduler=False):
        """
        Args:
        mode = 'train' | 'evaluate'
        inputs = {
            'branch': BranchInputsDataset(branch_inputs),
        }
        labels = {
            'coeff': CoeffLabelsDataset(coeff_labels) | None,
            'dm': ReducedDmLabelsDataset(dm_labels) | None,
        }
        batch_size = 8
        loss_weights = {
            'coeff': 1.0 | None,
            'dm': 1.0 | None,
            'dx': 1.0 | None
        }
        device = torch.device('cuda:3')
        debug = True
        shuffle = True
        return_outputs = False
        """
        if mode not in ['train', 'evaluate']:
            raise ValueError(f"Invalid mode: {mode}. Mode must be either 'train' or 'evaluate'.")

        avg_losses = {key: 0.0 for key in ['coeff', 'dm', 'dx', 'total']}
        num_functions = len(inputs['branch'])

        if batch_size == 'all' or batch_size > num_functions:
            batch_size = num_functions
            if debug:
                print(f'The batch size is set to {num_functions}')

        branch_indices_iterator = BatchIndicesIterator(0, len(inputs['branch']), batch_size, shuffle=shuffle)

        if return_outputs:
            outputs = {key: [] for key in ['coeff', 'dm']}

        self.model = self.model.to(device)
        self.model.train() if mode == 'train' else self.model.eval()
        start_time = time.time()
        for batched_branch_indices in branch_indices_iterator:
            batched_inputs = {'branch': inputs['branch'][batched_branch_indices, :].to(device)}
            keys = ['coeff', 'dm']
            batched_labels = {key: None for key in keys}
            batched_outputs = {key: None for key in keys}
            losses = {key: None for key in keys}
            for key in keys:
                if key  == 'coeff':
                    if labels[key] is not None and loss_weights[key] is not None and self.loss_functions[key] is not None:
                        batched_labels[key] = labels[key][batched_branch_indices,:].to(device)
                        batched_outputs[key] = self.model(batched_inputs['branch'])
                        losses[key] = self.loss_functions[key](batched_outputs[key], batched_labels[key])                      
                        avg_losses[key] += loss_weights[key] * losses[key].item()
                        if loss_weights['dx'] is not None and self.loss_functions['dx'] is not None:
                            losses['dx'] = self.loss_functions['dx'](batched_outputs[key], batched_labels[key])
                            avg_losses['dx'] += loss_weights['dx'] * losses['dx'].item()
                    else:
                        avg_losses[key] = None
                        avg_losses['dx'] = None

                if key == 'dm':
                    if labels[key] is not None and loss_weights[key] is not None and self.loss_functions[key] is not None:
                        batched_labels[key] = labels[key][batched_branch_indices,:,:].to(device)
                        batched_outputs[key] = self.batched_reduced_dm_functions(batched_inputs['branch'])  
                        losses[key] = self.loss_functions[key](batched_outputs[key], batched_labels[key])                      
                        avg_losses[key] += loss_weights[key] * losses[key].item()
                    else:
                        avg_losses[key] = None


            if all(value is None for value in losses.values()):
                raise ValueError("All values in 'losses' are None. Please check the arguments 'labels', 'loss_functions' and 'loss_weights'.")

            if mode == 'train':
                if self.loss_balancing_algorithm is not None:
                    self.loss_balancing_algorithm.rebalance(losses, loss_weights)

            total_loss = sum(loss_weights[key] * value for key, value in losses.items() if value is not None)
            avg_losses['total'] += total_loss.item()

            if mode == 'train':
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if return_outputs:
                for key, value in batched_outputs.items():
                    if value is not None:
                        outputs[key].append(value)
                    else:
                        outputs[key] = None

        if mode == 'train' and not disable_lr_scheduler:
            self.lr_scheduler.step()

        for key, value in avg_losses.items():
            if value is not None:
                avg_losses[key] /= num_functions

        if return_outputs:
            for key, value in outputs.items():
                if value is not None:
                    outputs[key] = torch.cat(value, dim=0)
                    if key == 'coeff':
                        outputs[key] = CoeffLabelsDataset(outputs[key])
                    elif key == 'dm':
                        outputs[key] = ReducedDmLabelsDataset(outputs[key])
                else:
                    outputs[key] = None

        end_time = time.time()
        if debug:
            print(f'Time [{mode}]: {format_elapsed_time(start_time, end_time)}')
            print(f'Losses [{mode}]: {avg_losses}')
            print(f'Loss weights: {loss_weights}')
            print('')

        if return_outputs:
            return avg_losses, outputs 
        return avg_losses


    def train(self, inputs, labels, batch_size, loss_weights, device, debug=True, shuffle=True, return_outputs=False, disable_lr_scheduler=False):
        return self._train_or_evaluate('train', inputs, labels, batch_size, loss_weights, device, debug, shuffle, return_outputs)

    def evaluate(self, inputs, labels, batch_size, loss_weights, device, debug=True, shuffle=False, return_outputs=False, disable_lr_scheduler=False):
        with torch.no_grad():
            return self._train_or_evaluate('evaluate', inputs, labels, batch_size, loss_weights, device, debug, shuffle, return_outputs)

    def post_process(self, outputs, labels, labels_stats):
        """
        Args:
        outputs = {
            'coeff': CoeffLabelsDataset(coeff_outputs) | None,
            'dm': ReducedDmLabelsDataset(dm_outputs) | None,
        }
        labels = {
            'coeff': CoeffLabelsDataset(coeff_labels) | None,
            'dm': ReducedDmLabelsDataset(dm_labels) | None,
        }
        labels_stats = {
            'mean': float,
            'std': float
        }
        Returns:
        outputs = {
            'coeff': numpy.ndarray | None,
            'dm': numpy.ndarray | None,
        }
        labels = {
            'coeff': numpy.ndarray | None,
            'dm': numpy.ndarray | None,
        }
        """
        def helper(data:  CoeffLabelsDataset | ReducedDmLabelsDataset | None, add_mean: bool=False)->numpy.ndarray | None:
            if data is not None:
                data = data[:].detach().cpu().numpy() if data[:].device.type == 'cuda' else data[:].numpy()
                data = data * labels_stats['std']
                if add_mean:
                    data += labels_stats['mean']
            return data
        outputs['coeff'] = helper(outputs['coeff'], add_mean=True)
        outputs['dm'] = helper(outputs['dm'])

        labels['coeff'] = helper(labels['coeff'], add_mean=True)
        labels['dm'] = helper(labels['dm'])
        return outputs, labels