import os
import sys
import argparse

import dolfin
import matplotlib  
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import Hyperelasticity # noqa
from utils import load_yaml, load_npy, save_npy, format_elapsed_time, timing # noqa

def hyperelasticity_plot_input_output(m: dolfin.Function, u:dolfin.Function) -> matplotlib.figure.Figure:
    figure = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    ax_m = dolfin.plot(m, mode='color', cmap='turbo', shading='gouraud')
    cbar_m = plt.colorbar(ax_m, pad=0.03, fraction=0.05, aspect=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_m.ax.tick_params(labelsize=12)
    plt.title(r'Parameter $m$', fontsize=16)

    plt.subplot(1, 2, 2)
    ax_u = dolfin.plot(u, mode='displacement', cmap='turbo', shading='gouraud')
    cbar_u = plt.colorbar(ax_u, pad=0.03, fraction=0.05, aspect=20)
    cbar_u.set_label(r'$||u||_2$', rotation=0, labelpad=15, fontsize=12)
    plt.title(r'Spatial point $x=X+u(X)$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar_u.ax.tick_params(labelsize=12)
    plt.xlabel(r'$X_1$', rotation=0, labelpad=0, fontsize=12)
    plt.ylabel(r'$X_2$', rotation=0, labelpad=10, fontsize=12)

    plt.subplots_adjust(wspace=0.0)  
    plt.close()

    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the input-output pairs in the hyperelasticity equation.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    args = parser.parse_args()
    dolfin.set_log_active(False)

    print(f'Running: {sys.argv[0]}')
    hyperelasticity = Hyperelasticity(load_yaml(args.mesh_config_path), load_yaml(args.function_space_config_path))
    num_figures = 10

    for dataset_path in [args.train_dataset_path, args.test_dataset_path]:
        input_nodal_values = load_npy(dataset_path + '/input_functions/nodal_values.npy')
        output_nodal_values = load_npy(dataset_path + '/output_functions/nodal_values.npy')

        for i in range(num_figures):
            m = dolfin.Function(hyperelasticity.Vh['parameter'])
            m.vector().set_local(input_nodal_values[i,:])
            u = dolfin.Function(hyperelasticity.Vh['state'])
            u.vector().set_local(output_nodal_values[i,:])
            figure = hyperelasticity_plot_input_output(m=m, u=u)
            figure.savefig(dataset_path + f'/figures/input_output_{i+1}.pdf', bbox_inches='tight')