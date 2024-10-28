import os
import sys
import argparse

import matplotlib
import matplotlib.pyplot as plt
import dolfin 

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import Hyperelasticity # noqa
from utils import load_yaml, load_npy, save_npy, timing # noqa


def plot_function(u_hat: dolfin.Function, title_1: str, title_2: str) -> matplotlib.figure.Figure:
    u_hat_1, u_hat_2 = u_hat.split(deepcopy=True)
    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    ax_1 = dolfin.plot(u_hat_1, mode='color', cmap='turbo', shading='gouraud')
    cbar_1 = plt.colorbar(ax_1, pad=0.03, fraction=0.05, aspect=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_1.ax.tick_params(labelsize=12)
    plt.title(title_1, fontsize=16)

    plt.subplot(1, 2, 2)
    ax_2 = dolfin.plot(u_hat_2, mode='color', cmap='turbo', shading='gouraud')
    cbar_2 = plt.colorbar(ax_2, pad=0.03, fraction=0.05, aspect=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_2.ax.tick_params(labelsize=12)
    plt.title(title_2, fontsize=16)

    plt.subplots_adjust(wspace=0.0)  
    plt.close()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the low rank approximation of output functions.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis.')
    parser.add_argument('--output_reduced_basis_name', type=str, choices=['ASM', 'POD', 'Random'], help='Name of the reduced basis of output function space.')
    parser.add_argument('--num_output_reduced_basis', type=int, help='Number of reduced basis of output function space.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    output_reduced_basis_path = args.output_reduced_basis_path
    output_reduced_basis_name = args.output_reduced_basis_name
    num_output_reduced_basis = args.num_output_reduced_basis

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]} with {num_output_reduced_basis} {output_reduced_basis_name} output reduced basis.') 

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)

    output_reduced_basis_nodal_values = load_npy(output_reduced_basis_path+f'/{output_reduced_basis_name}/nodal_values.npy')
    output_reduced_basis_nodal_values = output_reduced_basis_nodal_values[:num_output_reduced_basis,:]

    for dataset_path in [train_dataset_path, test_dataset_path]:
        reduced_outputs = load_npy(dataset_path+f'/reduced_outputs/{output_reduced_basis_name}.npy')
        reduced_outputs = reduced_outputs[:,:num_output_reduced_basis]
        low_rank_output_function_nodal_values = reduced_outputs @ output_reduced_basis_nodal_values
        save_npy(dataset_path+f'/low_rank_output_functions/{output_reduced_basis_name}_{num_output_reduced_basis}_nodal_values.npy', low_rank_output_function_nodal_values)
        for i in range(10):
            u_hat = dolfin.Function(hyperelasticity.Vh['state'])
            u_hat.vector().set_local(low_rank_output_function_nodal_values[i,:])
            fig = plot_function(u_hat, 
            title_1=f'$u^{{(low)}}_1$ (basis: {output_reduced_basis_name}; rank: {num_output_reduced_basis}; sample: {i+1})',
            title_2=f'$u^{{(low)}}_2$ (basis: {output_reduced_basis_name}; rank: {num_output_reduced_basis}; sample: {i+1})',
            )
            fig.savefig(dataset_path+f'/low_rank_output_functions/figures/{output_reduced_basis_name}_{num_output_reduced_basis}_{i+1}.pdf', bbox_inches='tight')
