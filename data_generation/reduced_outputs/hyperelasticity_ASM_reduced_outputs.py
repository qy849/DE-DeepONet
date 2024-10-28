import os
import sys
import argparse

import numpy 
import dolfin 
from mpi4py import MPI

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import Hyperelasticity # noqa
from utils import load_yaml, load_npy, save_npy, get_mass_matrix, get_stiffness_matrix, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ASM reduced outputs.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    output_reduced_basis_path = args.output_reduced_basis_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)

    ASM_basis = load_npy(output_reduced_basis_path+'/ASM/nodal_values.npy')
    num_reduced_basis = ASM_basis.shape[0]
    mass_matrix = get_mass_matrix(hyperelasticity.Vh['state'])
    for dataset_path in [train_dataset_path, test_dataset_path]:
        output_nodal_values = load_npy(dataset_path+'/output_functions/nodal_values.npy')
        num_functions = output_nodal_values.shape[0]
        temp_vector_1 = dolfin.Vector(MPI.COMM_SELF, hyperelasticity.Vh['state'].dim())
        temp_vector_2 = dolfin.Vector(MPI.COMM_SELF, hyperelasticity.Vh['state'].dim())
        temp_matrix = numpy.zeros((hyperelasticity.Vh['state'].dim(), num_reduced_basis))
        for i in range(num_reduced_basis):
            temp_vector_1[:] = ASM_basis[i,:]
            mass_matrix.mult(temp_vector_1, temp_vector_2)
            temp_matrix[:,i] = temp_vector_2.get_local()
        ASM_reduced_outputs = output_nodal_values @ temp_matrix
        save_npy(dataset_path+'/reduced_outputs/ASM.npy', ASM_reduced_outputs)

    print('Computing dx_loss_weighted_matrix for later use...')
    stiffness_matrix = get_stiffness_matrix(hyperelasticity.Vh['state'])
    dx_loss_weighted_matrix = ASM_basis @ (stiffness_matrix.array() @ ASM_basis.T)
    save_npy(output_reduced_basis_path+'/ASM/dx_loss_weighted_matrix.npy', dx_loss_weighted_matrix)