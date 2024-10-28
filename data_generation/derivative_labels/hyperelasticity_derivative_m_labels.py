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
from data_generation.operators import InputActiveOperator # noqa
from utils import load_yaml, load_npy, save_npy, load_and_scatter, gather_and_save, timing, format_elapsed_time # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the derivative m labels.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--basis_name', type=str, help='Name of the basis.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
    parser.add_argument('--samples_start_index', type=int, help='Start index of the samples to generate the derivative labels.')
    parser.add_argument('--samples_end_index', type=int, help='End index of the samples to generate the derivative labels.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    basis_name = args.basis_name
    dataset_path = args.dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path
    samples_start_index = args.samples_start_index
    samples_end_index = args.samples_end_index

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    input_dim = hyperelasticity.Vh['parameter'].dim()
    output_dim = hyperelasticity.Vh['state'].dim()

   
    input_reduced_basis_nodal_values = load_npy(input_reduced_basis_path+f'/{basis_name}/nodal_values.npy')    
    local_input_nodal_values, split = load_and_scatter(comm, dataset_path+'/input_functions/nodal_values.npy', start_index=samples_start_index, end_index=samples_end_index)
    local_output_nodal_values, split = load_and_scatter(comm, dataset_path+'/output_functions/nodal_values.npy', start_index=samples_start_index, end_index=samples_end_index)

    input_functions = []
    output_functions = []
    for i in range(split[rank]):
        input_function = dolfin.Function(hyperelasticity.Vh['parameter'])
        input_function.vector().set_local(local_input_nodal_values[i,:])
        input_functions.append(input_function)
        output_function = dolfin.Function(hyperelasticity.Vh['state'])
        output_function.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(output_function)

        
    num_input_reduced_basis = input_reduced_basis_nodal_values.shape[0]
    input_reduced_basis_functions = []
    for i in range(num_input_reduced_basis):
        input_reduced_basis_function = dolfin.Function(hyperelasticity.Vh['parameter'])
        input_reduced_basis_function.vector().set_local(input_reduced_basis_nodal_values[i,:])
        input_reduced_basis_functions.append(input_reduced_basis_function)


    local_derivative_labels = numpy.zeros((split[rank], hyperelasticity.mesh.num_vertices(), num_input_reduced_basis, 2))

    input_active_operator = InputActiveOperator(m_list=input_functions,
                                                u_list=output_functions,
                                                compile_form=hyperelasticity.compile_form,
                                                bcs0=hyperelasticity.bcs0,
                                                prior=GRF.prior)

    if rank == 0:
        start_time = MPI.Wtime()
    for i in range(split[rank]):
        derivative_functions = input_active_operator.jacobian_action(input_functions[i], output_functions[i], input_reduced_basis_functions)
        for j in range(num_input_reduced_basis):
            dm_1, dm_2 = derivative_functions[j].split(deepcopy=True)
            local_derivative_labels[i,:,j,0] = dm_1.compute_vertex_values(hyperelasticity.mesh)
            local_derivative_labels[i,:,j,1] = dm_2.compute_vertex_values(hyperelasticity.mesh)
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'Generating dm labels elapsed time (rank 0, {split[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    local_samples_start_index = samples_start_index + sum(split[:rank])
    local_samples_end_index = local_samples_start_index + split[rank]
    save_npy(dataset_path+f'/derivative_labels/{basis_name}_derivative_m_labels_{local_samples_start_index}_{local_samples_end_index}.npy', local_derivative_labels)