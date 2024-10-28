import os
import sys
import argparse
import numpy

import matplotlib
import matplotlib.pyplot as plt

import dolfin

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.probability_measure import GaussianRandomField # noqa
from utils import load_yaml, load_npy, format_elapsed_time, timing # noqa


def plot_eigenvalues(eigenvalues: numpy.ndarray, title: str) -> matplotlib.figure.Figure:

    indices = numpy.arange(1, len(eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(numpy.log10(indices), numpy.log10(eigenvalues), color='blue', marker='o', markersize=3)
    ax.set_xlabel(r'$\log_{10}(i)$', fontsize=15)
    ax.set_ylabel(r'$\log_{10}(\lambda_i)$', fontsize=15, rotation=0, labelpad=30)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=15) 
    plt.close()
    return fig 


def plot_eigenfunction(psi: dolfin.Function, title: str) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = dolfin.plot(psi, mode='color', cmap='turbo', shading='gouraud')
    plt.colorbar(ax, pad=0.03, fraction=0.05, aspect=20)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.close()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the input reduced basis.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis')
    args = parser.parse_args()
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')
    GRF = GaussianRandomField(load_yaml(args.mesh_config_path), load_yaml(args.function_space_config_path), load_yaml(args.gaussian_random_field_config_path))

    kle_basis_nodal_values = load_npy(input_reduced_basis_path+'/KLE/nodal_values.npy')
    kle_basis_eigenvalues = load_npy(input_reduced_basis_path+'/KLE/eigenvalues.npy')
    for i in range(kle_basis_nodal_values.shape[0]):
        psi = dolfin.Function(GRF.Vh['parameter'])
        psi.vector().set_local(kle_basis_nodal_values[i,:])
        fig = plot_eigenfunction(psi, title=f'KLE basis {i+1}')
        fig.savefig(input_reduced_basis_path+f'/KLE/figures/kle_basis_{i+1}.pdf', bbox_inches='tight')
    fig = plot_eigenvalues(kle_basis_eigenvalues, 'KLE eigenvalues')
    fig.savefig(input_reduced_basis_path+'/KLE/figures/kle_eigenvalues.pdf', bbox_inches='tight')

    active_basis_nodal_values = load_npy(input_reduced_basis_path+'/ASM/nodal_values.npy')
    active_basis_eigenvalues = load_npy(input_reduced_basis_path+'/ASM/eigenvalues.npy')
    for i in range(active_basis_nodal_values.shape[0]):
        psi = dolfin.Function(GRF.Vh['parameter'])
        psi.vector().set_local(active_basis_nodal_values[i,:])
        fig = plot_eigenfunction(psi, title=f'ASM basis {i+1}')
        fig.savefig(input_reduced_basis_path+f'/ASM/figures/asm_basis_{i+1}.pdf', bbox_inches='tight')
    fig = plot_eigenvalues(active_basis_eigenvalues, 'ASM eigenvalues')
    fig.savefig(input_reduced_basis_path+'/ASM/figures/asm_eigenvalues.pdf', bbox_inches='tight')

    random_basis_nodal_values = load_npy(input_reduced_basis_path+'/Random/nodal_values.npy')
    for i in range(random_basis_nodal_values.shape[0]):
        psi = dolfin.Function(GRF.Vh['parameter'])
        psi.vector().set_local(random_basis_nodal_values[i,:])
        fig = plot_eigenfunction(psi, title=f'Random basis {i+1}')
        fig.savefig(input_reduced_basis_path+f'/Random/figures/random_basis_{i+1}.pdf', bbox_inches='tight')