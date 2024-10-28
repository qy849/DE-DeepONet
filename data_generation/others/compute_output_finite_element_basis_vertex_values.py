import os
import sys
import time
import numpy
import argparse

import dolfin
from mpi4py import MPI # should be placed after "import dolfin" to avoid a potential bug 


repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, load_npy, save_npy, timing, format_elapsed_time, get_split, gather_and_save 
from data_generation.differential_equations import Hyperelasticity, NavierStokes 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the output finite element basis vertex values.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--save_path', type=str, help='Path to the directory saving the vertex values')
    parser.add_argument('--problem', type=str, choices=['hyperelasticity', 'navier_stokes'], help='The problem to solve.')

    args = parser.parse_args()
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    save_path = args.save_path
    problem = args.problem

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]}')


    if rank == 0:
        start_time = MPI.Wtime()
    mesh = dolfin.RectangleMesh(MPI.COMM_SELF,
                                dolfin.Point(0.0, 0.0), 
                                dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                mesh_args['num_x'], 
                                mesh_args['num_y'],
                                'right')

    if problem == 'hyperelasticity':
        hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
        num_vertices = mesh.num_vertices()
        num_finite_element_basis = hyperelasticity.Vh['state'].dim()
        if rank == 0:
            print(f'Number of vertices: {num_vertices}')
            print(f'Number of finite element basis: {num_finite_element_basis}')
        split_num_functions = get_split(num_finite_element_basis, size)
        local_output_finite_element_basis_vertex_values = numpy.zeros((split_num_functions[rank], num_vertices, 2))
        for i in range(split_num_functions[rank]):
            nodal_values = numpy.zeros(num_finite_element_basis)
            index = sum(split_num_functions[:rank]) + i
            nodal_values[index] = 1.0
            u = dolfin.Function(hyperelasticity.Vh['state'])
            u.vector().set_local(nodal_values)
            u1, u2 = u.split(deepcopy=True)
            local_output_finite_element_basis_vertex_values[i,:,0] = u1.compute_vertex_values(hyperelasticity.mesh)
            local_output_finite_element_basis_vertex_values[i,:,1] = u2.compute_vertex_values(hyperelasticity.mesh)

    elif problem == 'navier_stokes':
        navier_stokes = NavierStokes(mesh_args, function_space_args)
        num_vertices = mesh.num_vertices()
        num_finite_element_basis = navier_stokes.Vh['velocity'].dim()
        if rank == 0:
            print(f'Number of vertices: {num_vertices}')
            print(f'Number of finite element basis: {num_finite_element_basis}')
        split_num_functions = get_split(num_finite_element_basis, size)
        local_output_finite_element_basis_vertex_values = numpy.zeros((split_num_functions[rank], num_vertices, 2))
        matching_indices = [numpy.where(column)[0][0] for column in (navier_stokes.mesh.coordinates()[:, None, :] == mesh.coordinates()).all(axis=-1).T]
        for i in range(split_num_functions[rank]):
            nodal_values = numpy.zeros(num_finite_element_basis)
            index = sum(split_num_functions[:rank]) + i
            nodal_values[index] = 1.0
            u = dolfin.Function(navier_stokes.Vh['velocity'])
            u.vector().set_local(nodal_values)
            u1, u2 = u.split(deepcopy=True)
            local_output_finite_element_basis_vertex_values[i,:,0] = u1.compute_vertex_values(navier_stokes.mesh)[matching_indices]
            local_output_finite_element_basis_vertex_values[i,:,1] = u2.compute_vertex_values(navier_stokes.mesh)[matching_indices]
    
    gather_and_save(comm, save_path+'/output_finite_element_basis_vertex_values.npy', local_output_finite_element_basis_vertex_values, split_num_functions)
    if rank == 0: 
        end_time = MPI.Wtime()
        print(f'Elapsed time: {format_elapsed_time(start_time=start_time, end_time=end_time)}')