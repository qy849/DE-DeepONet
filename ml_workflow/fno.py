import os
import sys
import time
import math
import argparse

import torch
import numpy
from neuralop.models import FNO

import dolfin

# Append the repository path to the Python path
repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(repo_path)

from utils.loss import RelativeL2Loss  
from utils.debug import print_model_size, format_elapsed_time  
from utils.io import load_yaml, load_npy, load_pkl, save_pkl, save_yaml, save_csv, load_derivative_m_labels 
from utils.evaluator import Evaluator 
from utils.plot import plot_truth_pred_err
from utils.set_seed import set_seed 

from tuning.training_components import generate_training_components  
from training import LearningRateAnnealing, AddSpatialCoordinates 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the Fourier Neural Operator.')

    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')

    parser.add_argument('--config_models_path', type=str, help='Path to the models configuration directory.')
    parser.add_argument('--before_training_models_path', type=str, help='Path to the before training models directory.')
    parser.add_argument('--after_training_models_path', type=str, help='Path to the after training models directory.')

    parser.add_argument('--train_args_path', type=str, help='Path to the train arguments configuration file.')
    parser.add_argument('--update_train_args_path', type=str, help='Path to the updated train arguments configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset directory.')
    parser.add_argument('--train_dm_direction_path', type=str, help='Path to the train dm direction directory.')

    parser.add_argument('--test_args_path', type=str, help='Path to the test arguments configuration file.')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset directory.')
    parser.add_argument('--test_dm_path', type=str, help='Path to the test_dm directory.')

    parser.add_argument('--loss_and_error_path', type=str, help='Path to the loss and error directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    mesh_args = load_yaml(args.mesh_config_path)

    config_models_path = args.config_models_path
    before_training_models_path = args.before_training_models_path
    after_training_models_path = args.after_training_models_path
    train_dm_direction_path = args.train_dm_direction_path

    train_args_path = args.train_args_path
    update_train_args_path = args.update_train_args_path
    train_dataset_path = args.train_dataset_path

    test_args_path = args.test_args_path
    test_dataset_path = args.test_dataset_path
    test_dm_path = args.test_dm_path

    loss_and_error_path = args.loss_and_error_path
    seed = args.seed

    print(f'Random seed: {seed}')
    set_seed(seed=seed)
    if not os.path.exists(loss_and_error_path + f'/{seed}'):
        os.makedirs(loss_and_error_path + f'/{seed}')    
    print('')
    print('Constructing model...')
    models_filename = [f for f in os.listdir(config_models_path) if f.endswith('.yaml')]
    for filename in models_filename:
        models_args = load_yaml(os.path.join(config_models_path, filename))
        if models_args['in_channels'] != 3:
            raise ValueError('Only support FNO with input as a combination of a real-valued field and spatial coordinates in a 2D domain.')
        model = FNO(
            n_modes=(models_args['n_modes'], models_args['n_modes']), 
            hidden_channels=models_args['hidden_channels'], 
            in_channels=models_args['in_channels'], 
            out_channels=models_args['out_channels'],
            n_layers=models_args['n_layers'],
            )
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


        inputs = load_npy(train_dataset_path+"/input_functions/vertex_values.npy")[:train_args['num_train_functions']]
        labels = load_npy(train_dataset_path+"/output_functions/vertex_values.npy")[:train_args['num_train_functions']]
        labels_stats = {
            'mean': numpy.mean(labels), # type: float
            'std': numpy.std(labels) # type: float
        }
        labels = (labels - labels_stats['mean']) / labels_stats['std']

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
        labels = torch.from_numpy(labels).to(dtype=torch.float32)

        input_dim = (1,mesh_args['num_x']+1,mesh_args['num_y']+1)
        label_dim = (2,mesh_args['num_x']+1, mesh_args['num_y']+1)
        inputs = inputs.reshape(-1, *input_dim)
        labels = labels.permute(0,2,1).reshape(-1,*label_dim)

        loss_weights = {
            'eval': train_args['loss_weights.eval'],
            'dm': train_args['loss_weights.dm'],
        }   

        def compute_jvp(inputs_, direction_):
            return torch.func.jvp(model, (inputs_,), (direction_,))[1]
        compute_jvp_multi_directions = torch.vmap(compute_jvp, in_dims=(None, 0))
        batched_compute_jvp_multi_directions = torch.vmap(compute_jvp_multi_directions, in_dims=(0, None))


        dm_direction_vertex_values =  load_npy(train_dm_direction_path+f"/{train_args['dm_directions_name']}/vertex_values.npy")[:train_args['num_dm_directions'], :]
        dm_direction_vertex_values = torch.from_numpy(dm_direction_vertex_values).to(dtype=torch.float32)
        dm_direction_vertex_values = dm_direction_vertex_values.reshape(-1,*input_dim)

        # We find that adding spaital coordinates as additional inputs of FNO can greatly improve the model performance,
        # so for the dm direction, we add zero directions cooresponding to the spatial coordinates channels
        zero_directions = torch.zeros(train_args['num_dm_directions'],2,mesh_args['num_x']+1,mesh_args['num_y']+1) # 2 is the dimension of the spatial domain
        dm_direction_vertex_values = torch.cat((dm_direction_vertex_values, zero_directions), dim=1)
        dm_labels = load_derivative_m_labels(train_dataset_path+'/derivative_labels', train_args['dm_directions_name'])[:train_args['num_train_functions'],:,:train_args['num_dm_directions'],:]
        dm_labels = dm_labels / labels_stats['std']
        dm_labels = torch.from_numpy(dm_labels).to(dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(inputs,labels,dm_labels)
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_args['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
        )

        model, optimizer, lr_scheduler = generate_training_components(train_args, before_training_models_path)
        loss_func = RelativeL2Loss(reduction='sum') # unfortunately, the LpLoss in the neuralop package does not work when training the model with dm labels
        device = torch.device(train_args['device'])

        add_spatial_coordinates = AddSpatialCoordinates(mesh_args['num_x']+1, mesh_args['num_y']+1)

        model = model.to(device)
        dm_direction_vertex_values = dm_direction_vertex_values.to(device)
        loss_balancing_algorithm = LearningRateAnnealing(model, optimizer, update_frequency=train_args['update_frequency'], alpha=train_args['alpha']) 


        mesh = dolfin.RectangleMesh(dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], mesh_args['num_y'],'right')
        evaluator = Evaluator(mesh)

        test_args = load_yaml(test_args_path)
        test_inputs = load_npy(test_dataset_path+"/input_functions/vertex_values.npy")[:test_args['num_test_functions']]
        test_labels = load_npy(test_dataset_path+"/output_functions/vertex_values.npy")[:test_args['num_test_functions']]
        test_dm_labels = load_derivative_m_labels(test_dm_path)[:test_args['num_test_functions'],:,:test_args['num_test_dm_directions'],:]

        test_inputs = torch.from_numpy(test_inputs).to(dtype=torch.float32)
        test_inputs = test_inputs.reshape(-1, *input_dim)

        test_dm_direction_vertex_values = load_npy(test_dm_path+'/dm_direction_vertex_values.npy')[:test_args['num_test_dm_directions']]
        test_dm_direction_vertex_values = torch.from_numpy(test_dm_direction_vertex_values).to(dtype=torch.float32)
        test_dm_direction_vertex_values = test_dm_direction_vertex_values.reshape(-1,*input_dim)
        test_zero_directions = torch.zeros(test_args['num_test_dm_directions'],2,mesh_args['num_x']+1,mesh_args['num_y']+1)
        test_dm_direction_vertex_values = torch.cat((test_dm_direction_vertex_values, test_zero_directions), dim=1)
        test_dm_direction_vertex_values = test_dm_direction_vertex_values.to(device)

        test_dataset = torch.utils.data.TensorDataset(test_inputs)
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_args['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
        )

        test_avg_rel_l2_err_dict = {'u': []}
        test_avg_rel_h1_err_dict = {'u': []}
        test_avg_rel_fro_err_dict = {'dm': []}
        test_milestones = {'epoch': [], 'iteration': test_args['test_milestones']}

        all_epochs_start_time = time.time()
        iteration_counter = 0

        epochs = math.ceil(train_args['iterations'] / len(train_loader))
        stop_training_flag = False
        for epoch in range(1, epochs+1): 
            avg_losses = {key: 0.0 for key in ['eval', 'dm', 'total']} # for recording the average losses (across all samples, with equal weights) in each epoch 
            losses = {'eval': None,'dm': None} # for training (using one batched samples) in each iteration 
            one_epoch_start_time = time.time()
            samples_counter = 0
            for (batched_inputs, batched_labels, batched_dm_labels) in train_loader:
                model.train()
                samples_counter += len(batched_inputs)
                batched_inputs = add_spatial_coordinates(batched_inputs)
                batched_inputs, batched_labels, batched_dm_labels = batched_inputs.to(device), batched_labels.to(device), batched_dm_labels.to(device)
                batched_outputs = model(batched_inputs)
                losses['eval'] = loss_weights['eval']*torch.sum(loss_func(batched_outputs, batched_labels))
                avg_losses['eval'] += losses['eval'].item()
                """
                shape: 
                batched_inputs: (batch_size, 1+2, num_x+1, num_y+1)
                dm_direction_vertex_values: (num_dm_directions, 1, num_x+1, num_y+1)
                    unsqueeze(1) -> (num_dm_directions, 1, 1, num_x+1, num_y+1)
                    repeat(1, len(batched_inputs), 1, 1, 1): (num_dm_directions, batch_size, 1, num_x+1, num_y+1)
                batched_dm_outputs: (num_dm_directions, batch_size, output_dim, num_x+1, num_y+1)
                    reshape -> (num_dm_directions, batch_size, output_dim, (num_x+1)*(num_y+1))
                    permute -> (batch_size, (num_x+1)*(num_y+1), num_dm_directions, output_dim)
                """
                if train_args['loss_weights.dm'] is not None:
                    batched_dm_outputs = compute_jvp_multi_directions(batched_inputs, dm_direction_vertex_values.unsqueeze(1).repeat(1, len(batched_inputs), 1, 1, 1))
                    batched_dm_outputs = batched_dm_outputs.reshape(*batched_dm_outputs.shape[:3],-1)
                    batched_dm_outputs = batched_dm_outputs.permute(1,3,0,2)
                    losses['dm'] = loss_weights['dm'] * torch.sum(loss_func(batched_dm_outputs, batched_dm_labels))
                    avg_losses['dm'] += losses['dm'].item()
                                
                total_loss = sum(loss_weights[key] * value for key, value in losses.items() if value is not None)
                avg_losses['total'] += total_loss.item()
                loss_balancing_algorithm.rebalance(losses, loss_weights)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                iteration_counter += 1

                if iteration_counter in test_milestones['iteration']:
                    test_milestones['epoch'].append(epoch)
                    print(f'Testing the model at [epoch {epoch} | iteration {iteration_counter}]...')
                    print('Model inference on test dataset...')
                    test_outputs = []
                    test_dm_outputs = []

                    inference_start_time = {'eval': None, 'dm': None}
                    inference_time = {'eval': None, 'dm': None}
                    model.eval()
                    inference_start_time['eval'] = time.time()
                    with torch.no_grad():
                        for test_batched_inputs in test_loader:
                            test_batched_inputs = add_spatial_coordinates(test_batched_inputs[0])
                            test_batched_inputs = test_batched_inputs.to(device)
                            test_batched_outputs = model(test_batched_inputs)
                            test_outputs.append(test_batched_outputs)

                    test_outputs = torch.cat(test_outputs, dim=0)
                    test_outputs = test_outputs.permute(0,2,3,1).reshape(-1,label_dim[1]*label_dim[2],label_dim[0])
                    test_outputs = test_outputs*labels_stats['std'] + labels_stats['mean']
                    test_outputs = test_outputs.cpu().detach().numpy()
                    inference_time['eval'] = [time.time() - inference_start_time['eval']]

                    inference_start_time['dm'] = time.time()
                    with torch.no_grad():
                        for test_batched_inputs in test_loader:
                            test_batched_inputs = add_spatial_coordinates(test_batched_inputs[0])
                            test_batched_inputs = test_batched_inputs.to(device)
                            test_batched_dm_outputs = compute_jvp_multi_directions(test_batched_inputs, test_dm_direction_vertex_values.unsqueeze(1).repeat(1, len(test_batched_inputs), 1, 1, 1))
                            test_batched_dm_outputs = test_batched_dm_outputs.reshape(*test_batched_dm_outputs.shape[:3],-1)
                            test_batched_dm_outputs = test_batched_dm_outputs.permute(1,3,0,2)
                            test_dm_outputs.append(test_batched_dm_outputs)

                    test_dm_outputs = torch.cat(test_dm_outputs, dim=0)
                    test_dm_outputs = test_dm_outputs*labels_stats['std']
                    test_dm_outputs = test_dm_outputs.cpu().detach().numpy()
                    inference_time['dm'] = [time.time() - inference_start_time['dm']]
                    
                    print(f"Inference time (in seconds): {inference_time}")
                    print('')
                    save_csv(loss_and_error_path + f'/{seed}/inference_time_in_seconds.csv', inference_time)

                    print('Computing test errors...')
                    test_avg_rel_l2_err, test_avg_rel_h1_err, test_avg_rel_fro_err = evaluator.compute_multiple_avg_rel_err(
                        {'eval': test_outputs, 'dm': test_dm_outputs, 'dx': None}, 
                        {'eval': test_labels, 'dm': test_dm_labels, 'dx': None}
                    )
                    print(f'Test avg rel L2 err: {test_avg_rel_l2_err}')
                    print(f'Test avg rel H1 err: {test_avg_rel_h1_err}')
                    print(f'Test avg rel Fro err: {test_avg_rel_fro_err}')
                    print('')
                    test_avg_rel_l2_err_dict['u'].append(test_avg_rel_l2_err)
                    test_avg_rel_h1_err_dict['u'].append(test_avg_rel_h1_err)
                    test_avg_rel_fro_err_dict['dm'].append(test_avg_rel_fro_err['dm'])

                if iteration_counter == train_args['iterations']:
                    stop_training_flag = True
                    break

            for key, value in avg_losses.items():
                if value is not None:
                    avg_losses[key] /= samples_counter
            lr_scheduler.step()
            if train_args['debug']:
                print(f'**Epoch {epoch}**')
                print(f'Time: {format_elapsed_time(start_time=one_epoch_start_time, end_time=time.time())}')
                print(f'Losses: {avg_losses}')
                print(f'Loss weights: {loss_weights}')
                print('')
            if stop_training_flag:
                break

        print(f'Total epochs: {epochs}; Total iterations: {iteration_counter}')
        print(f'Total time (including multiple inference on test dataset): {format_elapsed_time(start_time=all_epochs_start_time, end_time=time.time())}')
        print(f'Final training losses: {avg_losses}')
        print('')

        model_name_seed = train_args['model']
        save_pkl(after_training_models_path + f'/pretrained_{model_name_seed}_{num_train_functions}.pkl', model)
        save_csv(loss_and_error_path + f'/{seed}/test_milestones_{num_train_functions}.csv', test_milestones)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_l2_error_{num_train_functions}.csv', test_avg_rel_l2_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_h1_error_{num_train_functions}.csv', test_avg_rel_h1_err_dict)
        save_csv(loss_and_error_path + f'/{seed}/test_avg_rel_fro_error_{num_train_functions}.csv', test_avg_rel_fro_err_dict)
        print('')

        print('Plotting figures...')
        coordinates = mesh.coordinates()
        X = coordinates[:,0].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
        Y = coordinates[:,1].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)

        num_plot_samples = 3
        output_dim = test_outputs.shape[2]
        for i in range(num_plot_samples):
            for j in range(output_dim):
                truth = test_labels[i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                pred =  test_outputs[i,:,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1) 
                error = abs(truth - pred)
                fig = plot_truth_pred_err(X, Y, truth, pred, error)
                fig.savefig(loss_and_error_path + f'/{seed}/truth_pred_error_{i+1}_{j+1}_{num_train_functions}.pdf', bbox_inches='tight')


        num_plot_directions = 16
        for i in range(num_plot_samples):
            for j in range(output_dim):
                for k in range(num_plot_directions):
                    dm_truth = test_dm_labels[i,:,k,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                    dm_pred = test_dm_outputs[i,:,k,j].reshape(mesh_args['num_x']+1, mesh_args['num_y']+1)
                    dm_error = abs(dm_truth-dm_pred)
                    fig = plot_truth_pred_err(X, Y, dm_truth, dm_pred, dm_error)
                    fig.savefig(loss_and_error_path + f'/{seed}/dm_truth_pred_error_{i+1}_{j+1}_{k+1}_{num_train_functions}.pdf', bbox_inches='tight')
        print('Done.')
        print('')