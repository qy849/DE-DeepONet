import os
import sys
import time
import math
import argparse

import numpy
import torch

import dolfin

# Append the repository path to the Python path
repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(repo_path)

from models import DeepONet, ScaledModel, StandardScaler 

from utils.loss import RelativeL2Loss 
from utils.debug import print_model_size, format_elapsed_time 
from utils.io import load_yaml, load_npy, load_pkl, save_pkl, save_yaml, save_csv, load_derivative_m_labels, load_derivative_x_labels  
from utils.evaluator import Evaluator  
from utils.plot import plot_truth_pred_err 
from utils.set_seed import set_seed

from tuning.model_utils import generate_nn
from tuning.training_components import generate_training_components

from training import BranchInputsDataset, TrunkInputsDataset, EvaluationLabelsDataset, DmLabelsDataset, DxLabelsDataset 
from training import LearningRateAnnealing, NoUpdate, Trainer 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the DeepONet which receives reduced inputs.')

    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')

    parser.add_argument('--config_models_path', type=str, help='Path to the models configuration directory.')
    parser.add_argument('--before_training_models_path', type=str, help='Path to the before training models directory.')
    parser.add_argument('--after_training_models_path', type=str, help='Path to the after training models directory.')

    parser.add_argument('--train_args_path', type=str, help='Path to the train arguments configuration file.')
    parser.add_argument('--update_train_args_path', type=str, help='Path to the updated train arguments configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset directory.')

    parser.add_argument('--test_args_path', type=str, help='Path to the test arguments configuration file.')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset directory.')
    parser.add_argument('--test_dm_path', type=str, help='Path to the test_dm directory.')

    parser.add_argument('--loss_and_error_path', type=str, help='Path to the loss and error directory.')
    parser.add_argument('--basis_name', type=str, choices=['ASM', 'KLE', 'Random'], help='The name of the input reduced basis.')
    parser.add_argument('--require_standardizing_inputs', action='store_true', default=False, help='Whether to standardize/z-score normalize branch inputs.')
    parser.add_argument('--require_positional_encoding', action='store_true', default=False, help='Whether to do positional encoding (of trunk inputs).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)

    config_models_path = args.config_models_path
    before_training_models_path = args.before_training_models_path
    after_training_models_path = args.after_training_models_path

    train_args_path = args.train_args_path
    update_train_args_path = args.update_train_args_path
    train_dataset_path = args.train_dataset_path

    test_args_path = args.test_args_path
    test_dataset_path = args.test_dataset_path
    test_dm_path = args.test_dm_path

    loss_and_error_path = args.loss_and_error_path
    basis_name = args.basis_name
    require_standardizing_inputs = args.require_standardizing_inputs
    require_positional_encoding = args.require_positional_encoding
    seed = args.seed

    print(f'Random seed: {seed}')
    set_seed(seed=seed)
    if not os.path.exists(loss_and_error_path + f'/{seed}'):
        os.makedirs(loss_and_error_path + f'/{seed}')    
    print('')
    print('Constructing model...')
    start_time = time.time()
    models_filename = [f for f in os.listdir(config_models_path) if f.endswith('.yaml')]
    for filename in models_filename:
        models_args = load_yaml(os.path.join(config_models_path, filename))
        branch = generate_nn(models_args['branch'])
        trunk = generate_nn(models_args['trunk'])
        model = DeepONet(branch, trunk, output_dim=models_args['output_dim'])
        model_name = models_args["name"]
        print(f'[{model_name}] ', end='')
        _,_, = print_model_size(model)
        save_pkl(os.path.join(before_training_models_path, f'{model_name}_{seed}.pkl'), model)

        # Positional encoding [random Fourier features]
        if require_positional_encoding:
            if models_args['positional_encoding']['output_dim'] % 2 != 0:
                raise ValueError('The output dimension of positional encoding must be even.')
            if models_args['trunk']['params']['input_dim'] != models_args['positional_encoding']['output_dim']:
                raise ValueError('The input dimension of positional encoding must be the same as the output dimension of trunk.')
            B = numpy.random.normal(loc=0.0, scale=models_args['positional_encoding']['sigma'], size=(int(models_args['positional_encoding']['output_dim']/2), int(models_args['positional_encoding']['input_dim'])))
            save_pkl(os.path.join(before_training_models_path, f'{model_name}_{seed}_positional_encoding.pkl'), B)

    print('Done.')
    print('')


    update_train_args = load_yaml(update_train_args_path)
    for num_train_functions in update_train_args['num_train_functions']:
        print(f'Training model with {num_train_functions} samples...')
        print('')

        train_args = load_yaml(train_args_path)
        train_args['model'] = f'{model_name}_{seed}'
        train_args['num_train_functions'] = num_train_functions
        train_args['iterations'] = update_train_args['iterations']
        train_args['debug'] = update_train_args['debug']
        save_yaml(train_args_path, train_args)

        branch_inputs = load_npy(train_dataset_path + f'/reduced_inputs/{basis_name}.npy')[:train_args['num_train_functions'],:train_args['num_reduced_basis']]
        if require_positional_encoding:
            coordinates = load_npy(train_dataset_path + '/output_functions/coordinates.npy') 
            trunk_inputs = numpy.concatenate((numpy.cos(2 * numpy.pi * (coordinates @ B.T)), numpy.sin(2 * numpy.pi * (coordinates @ B.T))), axis=1)
        else:
            trunk_inputs = load_npy(train_dataset_path + '/output_functions/coordinates.npy') 

        eval_labels = load_npy(train_dataset_path + '/output_functions/vertex_values.npy')[:train_args['num_train_functions'],:,:]
        dm_labels = load_derivative_m_labels(train_dataset_path+'/derivative_labels', basis_name)[:train_args['num_train_functions'],:,:train_args['num_reduced_basis'],:]
        dx_labels = load_derivative_x_labels(train_dataset_path+'/derivative_labels')[:train_args['num_train_functions'],:,:,:]

        labels_stats = {
            'mean': numpy.mean(eval_labels), # type: float
            'std': numpy.std(eval_labels) # type: float
        }

        eval_labels = (eval_labels - labels_stats['mean']) / labels_stats['std']
        dm_labels = dm_labels / labels_stats['std']
        dx_labels = dx_labels / labels_stats['std']
        
        train_inputs = {
            'branch': BranchInputsDataset(branch_inputs[:train_args['num_train_functions']]),
            'trunk': TrunkInputsDataset(trunk_inputs),
        }
        train_labels = {
            'eval': EvaluationLabelsDataset(eval_labels[:train_args['num_train_functions']]),
            'dm': DmLabelsDataset(dm_labels[:train_args['num_train_functions']]),
            'dx': DxLabelsDataset(dx_labels[:train_args['num_train_functions']]),
        }
        loss_functions = {
            'eval': RelativeL2Loss(reduction='sum'),
            'dm': RelativeL2Loss(reduction='sum'),
            'dx': RelativeL2Loss(reduction='sum')
        }
        loss_weights = {
            'eval': train_args['loss_weights.eval'],
            'dm': train_args['loss_weights.dm'],
            'dx': train_args['loss_weights.dx']
        }
        batch_size = {
            'branch': train_args['batch_size.branch'],
            'trunk': train_args['batch_size.trunk']
        }

        device = torch.device(train_args['device'])
        model, optimizer, lr_scheduler = generate_training_components(train_args, before_training_models_path)

        if require_standardizing_inputs:
            scaler = StandardScaler(train_inputs['branch'][:])
            model = ScaledModel(scaler, model)

        if 'update_frequency' in train_args.keys() and 'alpha' in train_args.keys():
            loss_balancing_algorithm = LearningRateAnnealing(model, optimizer, update_frequency=train_args['update_frequency'], alpha=train_args['alpha'])
        else:
            loss_balancing_algorithm = NoUpdate()

        mesh = dolfin.RectangleMesh(dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], 
                                    mesh_args['num_y'],
                                    'right')
        evaluator = Evaluator(mesh)

        test_args = load_yaml(test_args_path)
        if train_args['num_reduced_basis'] != test_args['num_reduced_basis']:
            raise ValueError(f'The number input reduced basis in training {train_args["num_reduced_basis"]} and testing {test_args["num_reduced_basis"]} are different.')

        test_branch_inputs = load_npy(test_dataset_path + f'/reduced_inputs/{basis_name}.npy')[:test_args['num_test_functions'],:test_args['num_reduced_basis']]
        if require_positional_encoding:
            coordinates = load_npy(test_dataset_path + '/output_functions/coordinates.npy') 
            test_trunk_inputs = numpy.concatenate((numpy.cos(2 * numpy.pi * (coordinates @ B.T)), numpy.sin(2 * numpy.pi * (coordinates @ B.T))), axis=1)
        else:
            test_trunk_inputs = load_npy(test_dataset_path + '/output_functions/coordinates.npy') 
        test_eval_labels = load_npy(test_dataset_path + '/output_functions/vertex_values.npy')[:test_args['num_test_functions'], :,:]
        test_dm_labels = load_derivative_m_labels(test_dataset_path + '/derivative_labels', basis_name)[:test_args['num_test_functions'],:,:test_args['num_reduced_basis'],:]
        test_dx_labels = load_derivative_x_labels(test_dataset_path + '/derivative_labels')[:test_args['num_test_functions'],:,:,:]

        test_dm_labels_random_directions = load_derivative_m_labels(test_dm_path)[:test_args['num_test_functions'],:,:test_args['num_test_dm_directions'],:]
        transformation_matrix = load_npy(test_dm_path + f'/{basis_name}_transformation_matrix.npy')[:test_args['num_reduced_basis'],:test_args['num_test_dm_directions']]

        test_eval_labels = (test_eval_labels - labels_stats['mean']) / labels_stats['std']
        test_dm_labels = test_dm_labels / labels_stats['std']
        test_dx_labels = test_dx_labels / labels_stats['std']

        test_avg_rel_l2_err_dict = {'u': []}
        test_avg_rel_h1_err_dict = {'u': []}
        test_avg_rel_fro_err_dict = {'dm': []}
        test_milestones = {'epoch': [], 'iteration': test_args['test_milestones']}

        if train_args['batch_size.trunk'] == 'all':
            train_args['batch_size.trunk'] = len(trunk_inputs)

        temp = math.ceil(math.ceil(train_args['sampling_fraction'] * len(trunk_inputs)) / train_args['batch_size.trunk'])
        iterations_per_epoch = math.ceil(train_args['num_train_functions'] / batch_size['branch']) * temp
        epochs = math.ceil(train_args['iterations'] / iterations_per_epoch)
        for iteration in test_milestones['iteration']:
            test_milestones['epoch'].append(math.ceil(iteration / iterations_per_epoch)) ## only approximately holds

        trainer = Trainer(model, optimizer, lr_scheduler, loss_functions, loss_balancing_algorithm=loss_balancing_algorithm)

        all_epochs_start_time = time.time()
        iteration_counter = 0
        for epoch in range(1, epochs+1): 
            if train_args['debug']:
                print(f'**Epoch {epoch}**')

            train_avg_losses = trainer.train(train_inputs, 
                                            train_labels, 
                                            batch_size, 
                                            train_args['sampling_fraction'], 
                                            loss_weights, 
                                            device, 
                                            train_args['debug'], 
                                            shuffle=True, 
                                            return_outputs=False, 
                                            disable_lr_scheduler=train_args['disable_lr_scheduler'])
            iteration_counter += iterations_per_epoch
            if epoch in test_milestones['epoch']:
                print(f'Testing the model at [epoch {epoch} | iteration {iteration_counter}]...')
                print('Model inference on test dataset...')
                test_inputs = {
                    'branch': BranchInputsDataset(test_branch_inputs),
                    'trunk': TrunkInputsDataset(test_trunk_inputs),
                }
                test_labels = {
                    'eval': EvaluationLabelsDataset(test_eval_labels),
                    'dm': DmLabelsDataset(test_dm_labels),
                    'dx': DxLabelsDataset(test_dx_labels),
                }
                test_loss_weights = {
                    'eval': test_args['loss_weights.eval'],
                    'dm': test_args['loss_weights.dm'],
                    'dx': test_args['loss_weights.dx']
                }
                test_batch_size = {
                    'branch': test_args['batch_size.branch'],
                    'trunk': test_args['batch_size.trunk']
                }
                inference_start_time = time.time()  
                test_avg_losses, test_outputs = trainer.evaluate(test_inputs, 
                                                                test_labels, 
                                                                batch_size=test_batch_size, 
                                                                sampling_fraction=1.0, 
                                                                loss_weights=test_loss_weights, 
                                                                device=device, 
                                                                debug=True,
                                                                shuffle=False, 
                                                                return_outputs=True,
                                                                disable_lr_scheduler=train_args['disable_lr_scheduler'])

                test_outputs, test_labels = trainer.post_process(test_outputs, test_labels, labels_stats)

                final_test_outputs = {'eval': None, 'dm': None, 'dx': None}
                final_test_outputs['eval'] =  test_outputs['eval']
                final_test_outputs['dm'] = torch.einsum('ijkl, km -> ijml', 
                                                        torch.tensor(test_outputs['dm'], device=device, dtype=torch.float64), 
                                                        torch.tensor(transformation_matrix, device=device, dtype=torch.float64)).cpu().numpy() 
                final_test_outputs['dx'] =  test_outputs['dx']

                # numpy version (slower)
                # final_test_outputs = {'eval': test_outputs['eval'], 'dm': numpy.einsum('ijkl, km -> ijml', test_outputs['dm'], transformation_matrix), 'dx': test_outputs['dx']}
                # final_test_labels = {'eval': test_labels['eval'], 'dm': test_dm_labels_random_directions, 'dx': test_labels['dx']}
                
                inference_time = {'eval and dm': [time.time() - inference_start_time]}
                print(f"Inference time (in seconds): {inference_time}")
                print('')
                save_csv(loss_and_error_path + f'/{seed}/inference_time_in_seconds.csv', inference_time)

                print('Computing test errors...')
                final_test_labels = {'eval': test_labels['eval'], 'dm': test_dm_labels_random_directions, 'dx': test_labels['dx']}
                test_avg_rel_l2_err, test_avg_rel_h1_err, test_avg_rel_fro_err = evaluator.compute_multiple_avg_rel_err(final_test_outputs, final_test_labels)       
                print(f'Test avg L2 err: {test_avg_rel_l2_err}')
                print(f'Test avg H1 err: {test_avg_rel_h1_err}')
                print(f'Test avg Fro err: {test_avg_rel_fro_err}')
                print('')
                test_avg_rel_l2_err_dict['u'].append(test_avg_rel_l2_err)
                test_avg_rel_h1_err_dict['u'].append(test_avg_rel_h1_err)
                test_avg_rel_fro_err_dict['dm'].append(test_avg_rel_fro_err['dm'])


        print(f'Total epochs: {epochs}; Total iterations: {iteration_counter}')
        print(f'Total time (including multiple inference on test dataset): {format_elapsed_time(start_time=all_epochs_start_time, end_time=time.time())}')
        print(f'Final training losses: {train_avg_losses}')
        print('')

        model_name_seed = train_args['model']
        save_pkl(after_training_models_path + f'/pretrained_{model_name_seed}_{basis_name}_{num_train_functions}.pkl', trainer.model)
        save_csv(loss_and_error_path + f'/{seed}/test_milestones_{num_train_functions}.csv', test_milestones)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_l2_error_{basis_name}_{num_train_functions}.csv', test_avg_rel_l2_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_h1_error_{basis_name}_{num_train_functions}.csv', test_avg_rel_h1_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_fro_error_{basis_name}_{num_train_functions}.csv', test_avg_rel_fro_err_dict)
        print('')

        print('Plotting figures...')
        coordinates = mesh.coordinates()
        X = coordinates[:,0].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
        Y = coordinates[:,1].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)

        num_plot_samples = 3
        output_dim = final_test_outputs['eval'].shape[2]
        for i in range(num_plot_samples):
            for j in range(output_dim):
                truth = final_test_labels['eval'][i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                pred =  final_test_outputs['eval'][i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1) 
                error = abs(truth - pred)
                fig = plot_truth_pred_err(X, Y, truth, pred, error)
                fig.savefig(loss_and_error_path + f'/{seed}/truth_pred_error_{i+1}_{j+1}_{num_train_functions}.pdf', bbox_inches='tight')


        num_plot_directions = 16
        if final_test_outputs['dm'] is not None:
            for i in range(num_plot_samples):
                for j in range(output_dim):
                    for k in range(num_plot_directions):
                        dm_truth = final_test_labels['dm'][i,:,k,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                        dm_pred = final_test_outputs['dm'][i,:,k,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                        dm_error = abs(dm_truth-dm_pred)
                        fig = plot_truth_pred_err(X, Y, dm_truth, dm_pred, dm_error)
                        fig.savefig(loss_and_error_path + f'/{seed}/dm_truth_pred_error_{i+1}_{j+1}_{k+1}_{num_train_functions}.pdf', bbox_inches='tight')
        print('Done.')
        print('')