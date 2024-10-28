import os
import sys
import time
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
    parser = argparse.ArgumentParser(description='Generate KLE reduced inputs .')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    dataset_path = args.dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')

    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    KLE_basis = load_npy(input_reduced_basis_path+'/KLE/nodal_values.npy')
    num_reduced_basis = KLE_basis.shape[0]
    temp_vector_1 = dolfin.Vector(MPI.COMM_SELF, GRF.Vh['parameter'].dim()) 
    temp_vector_2 = dolfin.Vector(MPI.COMM_SELF, GRF.Vh['parameter'].dim())
    input_nodal_values = load_npy(dataset_path+'/input_functions/nodal_values.npy')
    num_functions = input_nodal_values.shape[0]
    start_time = time.time()
    temp_matrix = numpy.zeros((GRF.Vh['parameter'].dim(), num_reduced_basis))
    for i in range(num_reduced_basis):
        temp_vector_1[:] = KLE_basis[i,:]
        GRF.prior.M.mult(temp_vector_1, temp_vector_2)
        temp_matrix[:,i] = temp_vector_2.get_local()

    KLE_reduced_inputs = input_nodal_values @ temp_matrix
    save_npy(dataset_path+'/reduced_inputs/KLE.npy', KLE_reduced_inputs)
