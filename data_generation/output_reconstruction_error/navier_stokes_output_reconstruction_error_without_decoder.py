import os
import sys
import argparse

import numpy 
import dolfin 
from mpi4py import MPI


repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import NavierStokes # noqa
from utils import load_yaml, load_npy, save_csv, timing # noqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute output reconstruction (relative L2) error via (finite element) solver without decoder.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
   
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    assert size == 2
    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    navier_stokes = NavierStokes(mesh_args, function_space_args)

    input_node_values = load_npy(train_dataset_path+'/input_functions/nodal_values.npy')
    output_node_values = load_npy(train_dataset_path+'/output_functions/velocity_nodal_values.npy')
    num_functions = input_node_values.shape[0]
    input_functions = []
    for i in range(num_functions):
        m = dolfin.Function(navier_stokes.Vh['parameter'])
        m.vector().set_local(input_node_values[i,:])
        input_functions.append(m)

    output_functions = []
    for i in range(num_functions):
        u = dolfin.Function(navier_stokes.Vh['velocity'])
        u.vector().set_local(output_node_values[i,:])
        output_functions.append(u)


    if rank == 0:
        ASM_nodal_values = load_npy(input_reduced_basis_path+'/ASM/nodal_values.npy')
        ASM_reduced_inputs = load_npy(train_dataset_path+'/reduced_inputs/ASM.npy')
        num_functions_for_error_estimation = int(num_functions/30)
        num_reduced_basis = ASM_nodal_values.shape[0]
        rank_array = numpy.around(numpy.geomspace(1, num_reduced_basis, num=20)).astype(int)
        unique_values, unique_indices = numpy.unique(rank_array, return_index=True)
        rank_array = rank_array[numpy.sort(unique_indices)]
        ASM_error = numpy.zeros((num_functions_for_error_estimation, len(rank_array)))
        for i,rank in enumerate(rank_array):
            low_rank_input_nodal_values = ASM_reduced_inputs[:,:rank] @ ASM_nodal_values[:rank,:]
            for j in range(num_functions_for_error_estimation):
                low_rank_input_function = dolfin.Function(navier_stokes.Vh['parameter'])
                low_rank_input_function.vector().set_local(low_rank_input_nodal_values[j,:])
                reconstructed_solution = navier_stokes.solve(m=low_rank_input_function)
                reconstructed_velocity, reconstructed_pressure = reconstructed_solution.split(deepcopy=True)
                ASM_error[j,i] = dolfin.errornorm(reconstructed_velocity, output_functions[j], norm_type='L2')/dolfin.norm(output_functions[j], norm_type='L2')
                
        ASM_mean_error = numpy.mean(ASM_error,axis=0)
        ASM_error_dict = {
            'rank': rank_array,
            'error': ASM_mean_error
        }
        save_csv(input_reduced_basis_path+'/ASM/output_reconstruction_error_without_decoder.csv', ASM_error_dict)


    if rank == 1:
        KLE_nodal_values = load_npy(input_reduced_basis_path+'/KLE/nodal_values.npy')
        KLE_reduced_inputs = load_npy(train_dataset_path+'/reduced_inputs/KLE.npy')

        num_functions_for_error_estimation = int(num_functions/30)
        num_reduced_basis = KLE_nodal_values.shape[0]
        rank_array = numpy.around(numpy.geomspace(1, num_reduced_basis, num=20)).astype(int)
        unique_values, unique_indices = numpy.unique(rank_array, return_index=True)
        rank_array = rank_array[numpy.sort(unique_indices)]
        KLE_error = numpy.zeros((num_functions_for_error_estimation, len(rank_array)))
        for i,rank in enumerate(rank_array):
            low_rank_input_nodal_values = KLE_reduced_inputs[:,:rank] @ KLE_nodal_values[:rank,:]
            for j in range(num_functions_for_error_estimation):
                low_rank_input_function = dolfin.Function(navier_stokes.Vh['parameter'])
                low_rank_input_function.vector().set_local(low_rank_input_nodal_values[j,:])
                reconstructed_solution = navier_stokes.solve(m=low_rank_input_function)
                reconstructed_velocity, reconstructed_pressure = reconstructed_solution.split(deepcopy=True)
                KLE_error[j,i] = dolfin.errornorm(reconstructed_velocity, output_functions[j], norm_type='L2')/dolfin.norm(output_functions[j], norm_type='L2')
    
        KLE_mean_error = numpy.mean(KLE_error,axis=0)
        KLE_error_dict = {
            'rank': rank_array,
            'error': KLE_mean_error
        }
        save_csv(input_reduced_basis_path+'/KLE/output_reconstruction_error_without_decoder.csv', KLE_error_dict)