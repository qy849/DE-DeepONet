import os
import sys
import argparse

import dolfin
import matplotlib  
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import NavierStokes # noqa
from utils import load_yaml, load_npy, save_npy, format_elapsed_time, timing # noqa



def navier_stokes_plot_input_output(m: dolfin.Function, u: dolfin.Function) -> matplotlib.figure.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(16, 8))

    plt.subplot(1, 3, 1)
    ax_m = dolfin.plot(m, mode='color', cmap='turbo', shading='gouraud')
    cbar_m = plt.colorbar(ax_m, pad=0.03, fraction=0.05, aspect=18.4)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_m.ax.tick_params(labelsize=12)
    plt.title(r'Parameter $m$', fontsize=16)

    velocity, pressure = u.split(deepcopy=True)
    velocity_x, velocity_y = velocity.split(deepcopy=True)

    plt.subplot(1, 3, 2)
    ax_u0 = dolfin.plot(velocity_x, mode= 'color', cmap='turbo', shading='gouraud')
    cbar_u0 = plt.colorbar(ax_u0, pad=0.03, fraction=0.05, aspect=18.4)
    plt.title(r'Velocity $x$-component', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_u0.ax.tick_params(labelsize=12)
    plt.xlabel(r'$x$', rotation=0, labelpad=0, fontsize=12)
    plt.ylabel(r'$y$', rotation=0, labelpad=10, fontsize=12)

    plt.subplot(1, 3, 3)
    ax_u1 = dolfin.plot(velocity_y, mode='color', cmap='turbo', shading='gouraud')
    cbar_u1 = plt.colorbar(ax_u1, pad=0.03, fraction=0.05, aspect=18.4)
    plt.title(r'Velocity $y$-component', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_u1.ax.tick_params(labelsize=12)
    plt.xlabel(r'$x$', rotation=0, labelpad=0, fontsize=12)
    plt.ylabel(r'$y$', rotation=0, labelpad=10, fontsize=12)

    plt.subplots_adjust(wspace=0.35)  
    plt.close()

    return figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the input-output pairs in the Navier--Stokes equations.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    dolfin.set_log_active(False)

    print(f'Running: {sys.argv[0]}')
    navier_stokes = NavierStokes(load_yaml(args.mesh_config_path), load_yaml(args.function_space_config_path))
    num_figures = 10

    dataset_path = args.dataset_path

    input_nodal_values = load_npy(dataset_path + '/input_functions/nodal_values.npy')
    output_nodal_values = load_npy(dataset_path + '/output_functions/nodal_values.npy')

    for i in range(num_figures):
        m = dolfin.Function(navier_stokes.Vh['parameter'])
        m.vector().set_local(input_nodal_values[i,:])
        u = dolfin.Function(navier_stokes.Vh['state'])
        u.vector().set_local(output_nodal_values[i,:])
        figure = navier_stokes_plot_input_output(m=m, u=u)
        figure.savefig(dataset_path + f'/figures/input_output_{i+1}.pdf', bbox_inches='tight')