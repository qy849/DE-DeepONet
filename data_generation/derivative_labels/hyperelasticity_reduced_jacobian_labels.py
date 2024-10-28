import os
import sys
import argparse

import dolfin 
from mpi4py import MPI

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.probability_measure import GaussianRandomField # noqa
from data_generation.differential_equations import Hyperelasticity # noqa
from data_generation.operators import InputActiveOperator # noqa
from utils import load_yaml, load_npy, load_and_scatter, gather_and_save, timing, format_elapsed_time # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ASM, KLE, and Random derivative m labels.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the reduced basis of input function space.')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the reduced basis of output function space.')
    parser.add_argument('--input_reduced_basis_name', type=str, choices=['ASM', 'KLE', 'Random'], help='Name of the reduced basis of input function space.')
    parser.add_argument('--output_reduced_basis_name', type=str, choices=['ASM', 'POD', 'Random'], help='Name of the reduced basis of output function space.')
    parser.add_argument('--num_output_reduced_basis', type=int, help='Number of reduced basis of output function space.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path
    output_reduced_basis_path = args.output_reduced_basis_path
    input_reduced_basis_name = args.input_reduced_basis_name
    output_reduced_basis_name = args.output_reduced_basis_name
    num_output_reduced_basis = args.num_output_reduced_basis

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    input_reduced_basis_nodal_values = load_npy(input_reduced_basis_path+f'/{input_reduced_basis_name}/nodal_values.npy')
    num_input_reduced_basis = input_reduced_basis_nodal_values.shape[0]
    output_reduced_basis_nodal_values = load_npy(output_reduced_basis_path+f'/{output_reduced_basis_name}/nodal_values.npy')
    output_reduced_basis_nodal_values = output_reduced_basis_nodal_values[:num_output_reduced_basis,:]

    for dataset_path in [train_dataset_path, test_dataset_path]:
        local_input_nodal_values, split = load_and_scatter(comm, dataset_path+'/input_functions/nodal_values.npy')
        local_low_rank_output_nodal_values, split = load_and_scatter(comm, dataset_path+f'/low_rank_output_functions/{output_reduced_basis_name}_{num_output_reduced_basis}_nodal_values.npy')

        input_functions = []
        for i in range(split[rank]):
            m = dolfin.Function(hyperelasticity.Vh['parameter'])
            m.vector().set_local(local_input_nodal_values[i,:])
            input_functions.append(m)

        low_rank_output_functions = []
        for i in range(split[rank]):
            u_hat = dolfin.Function(hyperelasticity.Vh['state'])
            u_hat.vector().set_local(local_low_rank_output_nodal_values[i,:])
            low_rank_output_functions.append(u_hat)

        input_active_operator = InputActiveOperator(m_list=input_functions,
                                                    u_list=low_rank_output_functions,
                                                    compile_form=hyperelasticity.compile_form,
                                                    bcs0=hyperelasticity.bcs0,
                                                    prior=GRF.prior)

        local_reduced_jacobian_labels = input_active_operator.compute_reduced_jacobian(input_reduced_basis_nodal_values, output_reduced_basis_nodal_values)
        gather_and_save(comm, dataset_path+f'/derivative_labels/{input_reduced_basis_name}_{output_reduced_basis_name}_{num_output_reduced_basis}_reduced_jacobian_labels.npy', local_reduced_jacobian_labels, split)
