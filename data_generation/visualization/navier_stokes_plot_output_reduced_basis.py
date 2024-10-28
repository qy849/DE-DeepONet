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

from data_generation.differential_equations import NavierStokes # noqa
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

def plot_eigenfunction(psi: dolfin.Function, title_1: str, title_2: str) -> matplotlib.figure.Figure:
    psi_1, psi_2 = psi.split(deepcopy=True)
    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    ax_1 = dolfin.plot(psi_1, mode='color', cmap='turbo', shading='gouraud')
    cbar_1 = plt.colorbar(ax_1, pad=0.03, fraction=0.05, aspect=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_1.ax.tick_params(labelsize=12)
    plt.title(title_1, fontsize=16)

    plt.subplot(1, 2, 2)
    ax_2 = dolfin.plot(psi_2, mode='color', cmap='turbo', shading='gouraud')
    cbar_2 = plt.colorbar(ax_2, pad=0.03, fraction=0.05, aspect=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_2.ax.tick_params(labelsize=12)
    plt.title(title_2, fontsize=16)

    plt.subplots_adjust(wspace=0.0)  
    plt.close()

    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the output reduced basis in the Navier--Stokes equations.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    output_reduced_basis_path = args.output_reduced_basis_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')
    navier_stokes = NavierStokes(mesh_args, function_space_args)

    Vh_velocity = navier_stokes.Vh['state'].sub(0).collapse()
   
    pod_basis_nodal_values = load_npy(output_reduced_basis_path+'/POD/nodal_values.npy')
    pod_basis_eigenvalues = load_npy(output_reduced_basis_path+'/POD/eigenvalues.npy')
    for i in range(pod_basis_nodal_values.shape[0]):
        psi = dolfin.Function(Vh_velocity)
        psi.vector().set_local(pod_basis_nodal_values[i,:])
        fig = plot_eigenfunction(psi, title_1=f'POD basis {i+1} ($x$-component)', title_2=f'POD basis {i+1} ($y$-component)')
        fig.savefig(output_reduced_basis_path+f'/POD/figures/pod_basis_{i+1}.pdf', bbox_inches='tight')
    fig = plot_eigenvalues(pod_basis_eigenvalues, 'POD eigenvalues')
    fig.savefig(output_reduced_basis_path+'/POD/figures/pod_eigenvalues.pdf', bbox_inches='tight')

    active_basis_nodal_values = load_npy(output_reduced_basis_path+'/ASM/nodal_values.npy')
    active_basis_eigenvalues = load_npy(output_reduced_basis_path+'/ASM/eigenvalues.npy')
    for i in range(active_basis_nodal_values.shape[0]):
        psi = dolfin.Function(Vh_velocity)
        psi.vector().set_local(active_basis_nodal_values[i,:])
        fig = plot_eigenfunction(psi, title_1=f'ASM basis {i+1} ($x$-component)', title_2=f'ASM basis {i+1} ($y$-component)')
        fig.savefig(output_reduced_basis_path+f'/ASM/figures/asm_basis_{i+1}.pdf', bbox_inches='tight')
    fig = plot_eigenvalues(active_basis_eigenvalues, 'ASM eigenvalues')
    fig.savefig(output_reduced_basis_path+'/ASM/figures/asm_eigenvalues.pdf', bbox_inches='tight')