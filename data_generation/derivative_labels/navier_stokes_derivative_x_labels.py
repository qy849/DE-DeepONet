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
from data_generation.differential_equations import NavierStokes # noqa
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

    navier_stokes = NavierStokes(mesh_args, function_space_args)

    num_vertices = navier_stokes.mesh.num_vertices()
    mesh_dim = navier_stokes.mesh.geometry().dim()
    vertex_coordinates = navier_stokes.mesh.coordinates()

    sub_mesh = dolfin.RectangleMesh(MPI.COMM_SELF,
                                    dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], 
                                    mesh_args['num_y'],
                                    'right')
    sub_vertex_coordinates = sub_mesh.coordinates()
    num_sub_vertices = sub_mesh.num_vertices()
    matching_indices = [numpy.where(column)[0][0] for column in (vertex_coordinates[:, None, :] == sub_vertex_coordinates).all(axis=-1).T]

    local_output_nodal_values, split = load_and_scatter(comm, dataset_path+'/output_functions/velocity_nodal_values.npy')

    output_functions = []
    for i in range(split[rank]):
        u = dolfin.Function(navier_stokes.Vh['velocity'])
        u.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(u)

    local_derivative_labels = numpy.zeros((split[rank], num_sub_vertices, mesh_dim, 2))
    Vh_grad = dolfin.VectorFunctionSpace(navier_stokes.mesh,  function_space_args['state']['u']['family'], function_space_args['state']['u']['degree'])
    
    if rank == 0:
        start_time = MPI.Wtime()
    for i in range(split[rank]):
        u = output_functions[i]
        u1, u2 = u.split(deepcopy=True)
        grad_u1 = dolfin.project(dolfin.grad(u1), Vh_grad)
        grad_u2 = dolfin.project(dolfin.grad(u2), Vh_grad)
        for j in range(num_sub_vertices):
            local_derivative_labels[i,j,:,0] = grad_u1(sub_vertex_coordinates[j,:])
            local_derivative_labels[i,j,:,1] = grad_u2(sub_vertex_coordinates[j,:])
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Generate dx labels elapsed time (rank 0, {split[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    local_samples_start_index = samples_start_index + sum(split[:rank])
    local_samples_end_index = local_samples_start_index + split[rank]

    save_npy(dataset_path+f'/derivative_labels/derivative_x_labels_{local_samples_start_index}_{local_samples_end_index}.npy', local_derivative_labels)