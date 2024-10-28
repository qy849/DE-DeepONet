import os
import sys
import argparse

import numpy 
import dolfin 
from mpi4py import MPI

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.probability_measure import GaussianRandomField # noqa
from data_generation.differential_equations import Hyperelasticity # noqa
from utils import load_yaml, load_npy, save_npy, load_and_scatter, gather_and_save, timing, format_elapsed_time # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the derivative x labels.')
    parser = argparse.ArgumentParser(description='Generate active subspace basis of input function space.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--samples_start_index', type=int, help='Start index of the samples to generate the derivative labels.')
    parser.add_argument('--samples_end_index', type=int, help='End index of the samples to generate the derivative labels.')
    
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    dataset_path = args.dataset_path
    samples_start_index = args.samples_start_index
    samples_end_index = args.samples_end_index

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)

    num_vertices = hyperelasticity.mesh.num_vertices()
    mesh_dim = hyperelasticity.mesh.geometry().dim()
    vertex_coordinates = hyperelasticity.mesh.coordinates()

    local_output_nodal_values, split = load_and_scatter(comm, dataset_path+'/output_functions/nodal_values.npy', start_index=samples_start_index, end_index=samples_end_index)

    output_functions = []
    for i in range(split[rank]):
        u = dolfin.Function(hyperelasticity.Vh['state'])
        u.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(u)

    local_derivative_labels = numpy.zeros((split[rank], num_vertices, mesh_dim, 2))
    Vh_grad = dolfin.VectorFunctionSpace(hyperelasticity.mesh,  function_space_args['state']['family'], function_space_args['state']['degree'])

    if rank == 0:
        start_time = MPI.Wtime()
    for i in range(split[rank]):
        u = output_functions[i]
        u1, u2 = u.split(deepcopy=True)
        grad_u1 = dolfin.project(dolfin.grad(u1), Vh_grad)
        grad_u2 = dolfin.project(dolfin.grad(u2), Vh_grad)
        for j in range(num_vertices):
            local_derivative_labels[i,j,:,0] = grad_u1(vertex_coordinates[j,:])
            local_derivative_labels[i,j,:,1] = grad_u2(vertex_coordinates[j,:])
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Generate dx labels elapsed time (rank 0, {split[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    local_samples_start_index = samples_start_index + sum(split[:rank])
    local_samples_end_index = local_samples_start_index + split[rank]

    save_npy(dataset_path+f'/derivative_labels/derivative_x_labels_{local_samples_start_index}_{local_samples_end_index}.npy', local_derivative_labels)