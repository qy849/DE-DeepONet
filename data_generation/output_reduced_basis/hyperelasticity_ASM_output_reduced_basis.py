import os
import sys
import argparse

import numpy 
import dolfin 
from mpi4py import MPI
import hippylib 

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.probability_measure import GaussianRandomField # noqa
from data_generation.differential_equations import Hyperelasticity # noqa
from data_generation.operators import OutputActiveOperator, AverageOperator # noqa
from utils import load_yaml, load_npy, save_npy, load_and_scatter, timing  # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate active subspace basis of output function space.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--output_reduced_basis_config_path', type=str, help='Path to the output reduced basis configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis.')
   
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    output_reduced_basis_args = load_yaml(args.output_reduced_basis_config_path)
    ASM_args = output_reduced_basis_args['ASM']
    train_dataset_path = args.train_dataset_path
    output_reduced_basis_path = args.output_reduced_basis_path

    dolfin.set_log_active(False)

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    local_input_nodal_values, split = load_and_scatter(comm, train_dataset_path+'/input_functions/nodal_values.npy', start_index=0, end_index=ASM_args['num_grad'])
    local_output_nodal_values, split = load_and_scatter(comm, train_dataset_path+'/output_functions/nodal_values.npy', start_index=0, end_index=ASM_args['num_grad'])

    random_test_matrix_shape = (ASM_args['num_reduced_basis'] + ASM_args['oversampling'], hyperelasticity.Vh['state'].dim())
    if rank == 0: 
        numpy.random.seed(ASM_args['seed'])
        temp_matrix = numpy.random.randn(*random_test_matrix_shape)
    else:
        temp_matrix = None

    temp_matrix = comm.bcast(temp_matrix, root=0)

    temp_dolfin_vector = dolfin.Vector(MPI.COMM_SELF, hyperelasticity.Vh['state'].dim())
    random_test_matrix = hippylib.MultiVector(temp_dolfin_vector, random_test_matrix_shape[0])
        
    for i in range(random_test_matrix_shape[0]):
        random_test_matrix[i][:] = temp_matrix[i,:]

    input_functions = []
    output_functions = []
    for i in range(split[rank]):
        input_function = dolfin.Function(hyperelasticity.Vh['parameter'])
        input_function.vector().set_local(local_input_nodal_values[i,:])
        input_functions.append(input_function)
        output_function = dolfin.Function(hyperelasticity.Vh['state'])
        output_function.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(output_function)

    output_active_operator = OutputActiveOperator(m_list=input_functions,
                                                  u_list=output_functions,
                                                  compile_form=hyperelasticity.compile_form,
                                                  bcs0=hyperelasticity.bcs0,
                                                  prior=GRF.prior)

    average_output_active_operator = AverageOperator(local_operator=output_active_operator, comm=comm)

    eigvals, eigvecs = hippylib.doublePassG(A=average_output_active_operator,
                                            B=output_active_operator.M['state'],
                                            Binv=output_active_operator.Msolver['state'],
                                            Omega=random_test_matrix, 
                                            k=ASM_args['num_reduced_basis'], 
                                            s=1, 
                                            check = False)

    if rank == 0: 
        active_basis_nodal_values = numpy.zeros((ASM_args['num_reduced_basis'], hyperelasticity.Vh['state'].dim()))
        for i in range(ASM_args['num_reduced_basis']):
            active_basis_nodal_values[i,:] = eigvecs[i].get_local()
        save_npy(output_reduced_basis_path+'/ASM/eigenvalues.npy', eigvals[:ASM_args['num_reduced_basis']])
        save_npy(output_reduced_basis_path+'/ASM/nodal_values.npy', active_basis_nodal_values)    




