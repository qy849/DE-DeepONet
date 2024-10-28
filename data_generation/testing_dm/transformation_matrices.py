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
from utils import load_yaml, load_npy, save_npy, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the transformation matrices for computing the derivative m outputs (reduced without decoder).')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
    parser.add_argument('--test_dm_path', type=str, help='Path to the test_dm directory')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    input_reduced_basis_path = args.input_reduced_basis_path
    test_dm_path = args.test_dm_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')
    
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    dm_direction_nodal_values = load_npy(test_dm_path+'/dm_direction_nodal_values.npy')
    num_directions = dm_direction_nodal_values.shape[0]
    temp_vector_1 = dolfin.Vector(MPI.COMM_SELF, GRF.Vh['parameter'].dim())
    temp_vector_2 = dolfin.Vector(MPI.COMM_SELF, GRF.Vh['parameter'].dim())

    ASM_basis = load_npy(input_reduced_basis_path+'/ASM/nodal_values.npy')
    KLE_basis = load_npy(input_reduced_basis_path+'/KLE/nodal_values.npy')

    temp_matrix = numpy.zeros((GRF.Vh['parameter'].dim(), num_directions))
    for i in range(num_directions):
        temp_vector_1[:] = dm_direction_nodal_values[i,:]
        GRF.prior.R.mult(temp_vector_1, temp_vector_2)
        temp_matrix[:,i] = temp_vector_2.get_local()
    ASM_transformation_matrix =  ASM_basis @ temp_matrix 
    save_npy(test_dm_path+'/ASM_transformation_matrix.npy', ASM_transformation_matrix)


    temp_matrix = numpy.zeros((GRF.Vh['parameter'].dim(), num_directions))
    for i in range(num_directions):
        temp_vector_1[:] = dm_direction_nodal_values[i,:]
        GRF.prior.M.mult(temp_vector_1, temp_vector_2)
        temp_matrix[:,i] = temp_vector_2.get_local()
    KLE_transformation_matrix =  KLE_basis @ temp_matrix 
    save_npy(test_dm_path+'/KLE_transformation_matrix.npy', KLE_transformation_matrix)

