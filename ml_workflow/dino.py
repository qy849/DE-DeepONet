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

from models import ScaledModel_v2, StandardScaler 

from utils.loss import RelativeL2Loss, RelativeWeightedL2Loss 
from utils.debug import print_model_size, format_elapsed_time 
from utils.io import load_yaml, load_npy, load_pkl, save_pkl, save_yaml, save_csv, load_derivative_m_labels, load_derivative_x_labels 
from utils.evaluator import Evaluator 
from utils.plot import plot_truth_pred_err   
from utils.set_seed import set_seed

from tuning.model_utils import generate_nn
from tuning.training_components import generate_training_components 

from training import BranchInputsDataset, CoeffLabelsDataset, ReducedDmLabelsDataset 
from training import LearningRateAnnealing, NoUpdate, Trainer_v2 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the POD(or ASM)-DeepONet which receives reduced inputs.')

    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file')

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
    parser.add_argument('--input_reduced_basis_name', type=str, choices=['ASM', 'KLE', 'Random'], help='The name of the input reduced basis.')
    parser.add_argument('--output_reduced_basis_name', type=str, choices=['ASM', 'POD', 'Random'], help='The name of the output reduced basis.')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis.')
    parser.add_argument('--require_standardizing_inputs', action='store_true', default=False, help='Whether to standardize/z-score normalize branch inputs.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)

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
    input_reduced_basis_name = args.input_reduced_basis_name
    output_reduced_basis_name = args.output_reduced_basis_name
    output_reduced_basis_path = args.output_reduced_basis_path
    require_standardizing_inputs = args.require_standardizing_inputs
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
        model = generate_nn(models_args['branch'])
        model_name = models_args["name"]
        print(f'[{model_name}] ', end='')
        _,_, = print_model_size(model)
        save_pkl(os.path.join(before_training_models_path, f'{model_name}_{seed}.pkl'), model)

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

        branch_inputs = load_npy(train_dataset_path + f'/reduced_inputs/{input_reduced_basis_name}.npy')[:train_args['num_train_functions'],:train_args['num_input_reduced_basis']]
        coeff_labels = load_npy(train_dataset_path + f'/reduced_outputs/{output_reduced_basis_name}.npy')[:train_args['num_train_functions'],:train_args['num_output_reduced_basis']]
        dm_labels = load_npy(train_dataset_path + f'/derivative_labels/{input_reduced_basis_name}_{output_reduced_basis_name}_{train_args["num_output_reduced_basis"]}_reduced_jacobian_labels.npy')
        dm_labels = dm_labels[:train_args['num_train_functions'],:,:train_args['num_input_reduced_basis']]
    
        if dm_labels.shape[1] != coeff_labels.shape[1]:
            raise ValueError(f'The rank of the low rank output function in derivative m labels {dm_labels.shape[1]} is not equal to the rank of the low rank output function in coeff labels {coeff_labels.shape[1]}.')

        labels_stats = {
            'mean': numpy.mean(coeff_labels), # type: float
            'std': numpy.std(coeff_labels) # type: float
        }

        coeff_labels = (coeff_labels - labels_stats['mean']) / labels_stats['std']
        dm_labels = dm_labels / labels_stats['std']

        train_inputs = {
            'branch': BranchInputsDataset(branch_inputs[:train_args['num_train_functions']])
        }

        train_labels = {
            'coeff': CoeffLabelsDataset(coeff_labels[:train_args['num_train_functions']]),
            'dm': ReducedDmLabelsDataset(dm_labels[:train_args['num_train_functions']])
        }
        dx_loss_weighted_matrix = load_npy(output_reduced_basis_path + f'/{output_reduced_basis_name}/dx_loss_weighted_matrix.npy')[:train_args['num_output_reduced_basis'],:train_args['num_output_reduced_basis']]
        dx_loss_weighted_matrix = torch.from_numpy(dx_loss_weighted_matrix).to(dtype=torch.float32)
        relative_weighted_l2_loss = RelativeWeightedL2Loss(dx_loss_weighted_matrix, reduction='sum')
        device = torch.device(train_args['device'])
        relative_weighted_l2_loss = relative_weighted_l2_loss.to(device)
        loss_functions = {
            'coeff': RelativeL2Loss(reduction='sum'),
            'dm': RelativeL2Loss(reduction='sum'),
            'dx': relative_weighted_l2_loss
        }
        loss_weights = {
            'coeff': train_args['loss_weights.coeff'],
            'dm': train_args['loss_weights.dm'],
            'dx': train_args['loss_weights.dx']
        }

        model, optimizer, lr_scheduler = generate_training_components(train_args, before_training_models_path)

        if require_standardizing_inputs:
            scaler = StandardScaler(train_inputs['branch'][:])
            model = ScaledModel_v2(scaler, model)

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
        if train_args['num_input_reduced_basis'] != test_args['num_input_reduced_basis']: 
            raise ValueError(f'The number of input reduced basis in training {train_args["num_input_reduced_basis"]} and testing {test_args["num_input_reduced_basis"]} are different.')

        test_branch_inputs = load_npy(test_dataset_path + f'/reduced_inputs/{input_reduced_basis_name}.npy')
        test_coeff_labels = load_npy(test_dataset_path + f'/reduced_outputs/{output_reduced_basis_name}.npy')
        test_dm_labels = load_npy(test_dataset_path + f'/derivative_labels/{input_reduced_basis_name}_{output_reduced_basis_name}_{train_args["num_output_reduced_basis"]}_reduced_jacobian_labels.npy')[:test_args['num_test_functions']]

        test_eval_labels = load_npy(test_dataset_path + '/output_functions/vertex_values.npy')[:test_args['num_test_functions'], :,:]
        test_dm_labels_random_directions = load_derivative_m_labels(test_dm_path)[:test_args['num_test_functions'],:,:test_args['num_test_dm_directions'],:]

        output_reduced_basis_nodal_values = load_npy(output_reduced_basis_path + f'/{output_reduced_basis_name}/nodal_values.npy')[:train_args['num_output_reduced_basis'],:]
        output_finite_element_basis_vertex_values = load_npy(output_reduced_basis_path + '/output_finite_element_basis_vertex_values.npy')
        transformation_matrix = load_npy(test_dm_path + f'/{input_reduced_basis_name}_transformation_matrix.npy')[:test_args['num_input_reduced_basis'],:test_args['num_test_dm_directions']]

        test_branch_inputs = test_branch_inputs[:test_args['num_test_functions'],:test_args['num_input_reduced_basis']]
        test_coeff_labels = test_coeff_labels[:test_args['num_test_functions'],:test_args['num_output_reduced_basis']]
        test_dm_labels = test_dm_labels[:test_args['num_test_functions'],:,:test_args['num_input_reduced_basis']]

        if test_dm_labels.shape[1] != test_coeff_labels.shape[1]:
            raise ValueError(f'The rank of the low rank output function in derivative m labels {test_dm_labels.shape[1]} is not equal to the rank of the low rank output function in coeff labels {test_coeff_labels.shape[1]}.')

        test_coeff_labels = (test_coeff_labels - labels_stats['mean']) / labels_stats['std']
        test_dm_labels = test_dm_labels / labels_stats['std']
        
        iterations_per_epoch = math.ceil(train_args['num_train_functions'] / train_args['batch_size'])
        epochs = math.ceil(train_args['iterations'] / iterations_per_epoch)

        test_avg_rel_l2_err_dict = {'u': []}
        test_avg_rel_h1_err_dict = {'u': []}
        test_avg_rel_fro_err_dict = {'dm': []}
        test_milestones = {'epoch': [], 'iteration': test_args['test_milestones']}
        for iteration in test_milestones['iteration']:
            test_milestones['epoch'].append(math.ceil(iteration / iterations_per_epoch)) ## only approximately holds

        trainer = Trainer_v2(model, optimizer, lr_scheduler, loss_functions, loss_balancing_algorithm=loss_balancing_algorithm)

        all_epochs_start_time = time.time()
        iteration_counter = 0
        for epoch in range(1, epochs+1):
            if train_args['debug']:
                print(f'**Epoch {epoch}**')
            train_avg_losses = trainer.train(train_inputs, 
                                                train_labels, 
                                                batch_size=train_args['batch_size'], 
                                                loss_weights=loss_weights, 
                                                device=device, 
                                                debug=train_args['debug'], 
                                                shuffle=True, 
                                                return_outputs=False,
                                                disable_lr_scheduler=train_args['disable_lr_scheduler'])           
            iteration_counter += iterations_per_epoch
            if epoch in test_milestones['epoch']:
                print(f'Testing the model at [epoch {epoch} | iteration {iteration_counter}]...')
                print('Model inference on test dataset...')
                print('')
                test_inputs = {
                    'branch': BranchInputsDataset(test_branch_inputs)
                }
                test_labels = {
                    'coeff': CoeffLabelsDataset(test_coeff_labels),
                    'dm': ReducedDmLabelsDataset(test_dm_labels)
                }
                test_loss_weights = {
                    'coeff': test_args['loss_weights.coeff'],
                    'dm': test_args['loss_weights.dm'],
                    'dx': test_args['loss_weights.dx']
                }

                inference_start_time = time.time()  
                test_avg_losses, test_outputs = trainer.evaluate(test_inputs, 
                                                                    test_labels, 
                                                                    batch_size=test_args['batch_size'],
                                                                    loss_weights=test_loss_weights,
                                                                    device=device, 
                                                                    debug=True, 
                                                                    shuffle=False,
                                                                    return_outputs=True,
                                                                    disable_lr_scheduler=train_args['disable_lr_scheduler'])
                test_outputs, test_labels = trainer.post_process(test_outputs, test_labels, labels_stats)
    
                final_test_outputs = {'eval': None, 'dm': None, 'dx': None}
                left_temp_mtx = torch.einsum('ijk, li -> ljk', 
                                            torch.tensor(output_finite_element_basis_vertex_values, device=device, dtype=torch.float64), 
                                            torch.tensor(output_reduced_basis_nodal_values, device=device, dtype=torch.float64))

                final_test_outputs['eval'] = torch.einsum('ij, jlm -> ilm', 
                                                            torch.tensor(test_outputs['coeff'], device=device, dtype=torch.float64), 
                                                            left_temp_mtx).cpu().numpy()

                right_temp_mtx = torch.einsum('ijk, km -> ijm', 
                                                torch.tensor(test_outputs['dm'], device=device, dtype=torch.float64),
                                                torch.tensor(transformation_matrix, device=device, dtype=torch.float64))
                final_test_outputs['dm'] = torch.einsum('ijk, jmn -> imkn', right_temp_mtx, left_temp_mtx).cpu().numpy()

                # numpy version (extremely slow)
                # left_temp_mtx = numpy.einsum('ijk, li -> ljk', output_finite_element_basis_vertex_values, output_reduced_basis_nodal_values)
                # final_test_outputs['eval'] = numpy.einsum('ij, jlm -> ilm', test_outputs['coeff'], left_temp_mtx)

                # right_temp_mtx = numpy.einsum('ijk, km -> ijm', test_outputs['dm'], transformation_matrix)
                # final_test_outputs['dm'] = numpy.einsum('ijk, jmn -> imkn', right_temp_mtx, left_temp_mtx) # bottleneck, much slower than torch version

                inference_time = {'eval and dm': [time.time() - inference_start_time]}
                print(f"Inference time (in seconds) (excluding evaluation of output finite element basis functions): {inference_time}") # evaluations are given in the data generation part
                print('')
                save_csv(loss_and_error_path + f'/{seed}/inference_time_in_seconds.csv', inference_time)

                print('Computing test error...')
                final_test_labels = {'eval': test_eval_labels, 'dm': test_dm_labels_random_directions, 'dx': None}
                test_avg_rel_l2_err, test_avg_rel_h1_err, test_avg_rel_fro_err = evaluator.compute_multiple_avg_rel_err(final_test_outputs, final_test_labels)

                print(f'Average losses: {test_avg_losses}')
                print(f'Test avg rel L2 err: {test_avg_rel_l2_err}')
                print(f'Test avg rel H1 err: {test_avg_rel_h1_err}')
                print(f'Test avg rel Fro  err: {test_avg_rel_fro_err}')
                print('')
                test_avg_rel_l2_err_dict['u'].append(test_avg_rel_l2_err)
                test_avg_rel_h1_err_dict['u'].append(test_avg_rel_h1_err)
                test_avg_rel_fro_err_dict['dm'].append(test_avg_rel_fro_err['dm'])


        print(f'Total epochs: {epochs}; Total iterations: {iteration_counter}')
        print(f'Total time (including multiple inference on test dataset): {format_elapsed_time(start_time=all_epochs_start_time, end_time=time.time())}')
        print(f'Final training losses: {train_avg_losses}')
        print('')

        model_name_seed = train_args['model']
        save_pkl(after_training_models_path + f'/pretrained_{model_name_seed}_{num_train_functions}.pkl', trainer.model)
        save_csv(loss_and_error_path + f'/{seed}/test_milestones_{num_train_functions}.csv', test_milestones)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_l2_error_{num_train_functions}.csv', test_avg_rel_l2_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_h1_error_{num_train_functions}.csv', test_avg_rel_h1_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_fro_error_{num_train_functions}.csv',test_avg_rel_fro_err_dict)
    
        print('Plotting figures...')
        coordinates = mesh.coordinates()
        X = coordinates[:,0].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
        Y = coordinates[:,1].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)

        num_plot_samples = 3
        output_dim = final_test_outputs['eval'].shape[2]    
        for i in range(num_plot_samples):
            for j in range(output_dim):
                truth = final_test_labels['eval'][i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                pred = final_test_outputs['eval'][i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
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
