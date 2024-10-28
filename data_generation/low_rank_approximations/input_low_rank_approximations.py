import os
import sys
import argparse

import dolfin
import matplotlib
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.probability_measure import GaussianRandomField # noqa
from utils import load_yaml, load_npy, save_npy, timing # noqa


def plot_function(m_hat: dolfin.Function, title: str) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = dolfin.plot(m_hat, mode='color', cmap='turbo', shading='gouraud')
    plt.colorbar(ax, pad=0.03, fraction=0.05, aspect=20)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.close()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the low rank approximation of input functions.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
    parser.add_argument('--input_reduced_basis_name', type=str, choices=['ASM', 'KLE', 'Random'], help='Name of the reduced basis of input function space.')
    parser.add_argument('--num_input_reduced_basis', type=int, help='Number of reduced basis of input function space.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    dataset_path = args.dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path
    input_reduced_basis_name = args.input_reduced_basis_name
    num_input_reduced_basis = args.num_input_reduced_basis

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]} with {num_input_reduced_basis} {input_reduced_basis_name} input reduced basis.')

    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    input_reduced_basis_nodal_values = load_npy(input_reduced_basis_path+f'/{input_reduced_basis_name}/nodal_values.npy')
    input_reduced_basis_nodal_values =  input_reduced_basis_nodal_values[:num_input_reduced_basis,:]
    reduced_inputs = load_npy(dataset_path+f'/reduced_inputs/{input_reduced_basis_name}.npy')
    reduced_inputs = reduced_inputs[:,:num_input_reduced_basis]
    low_rank_input_function_nodal_values = reduced_inputs @ input_reduced_basis_nodal_values
    save_npy(dataset_path+f'/low_rank_input_functions/{input_reduced_basis_name}_{num_input_reduced_basis}_nodal_values.npy', low_rank_input_function_nodal_values)
    for i in range(10):
        m_hat = dolfin.Function(GRF.Vh['parameter'])
        m_hat.vector().set_local(low_rank_input_function_nodal_values[i,:])
        fig = plot_function(m_hat, title=f'$m^{{(low)}}$ (basis: {input_reduced_basis_name}; rank: {num_input_reduced_basis}; sample: {i+1})')
        fig.savefig(dataset_path+f'/low_rank_input_functions/figures/{input_reduced_basis_name}_{num_input_reduced_basis}_{i+1}.pdf', bbox_inches='tight')