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

from data_generation.operators import KLEOperator # noqa
from data_generation.probability_measure import GaussianRandomField # noqa
from utils import load_yaml, load_npy, save_npy, get_nodal_values, timing, format_elapsed_time # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the KLE basis of input function space.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--input_reduced_basis_config_path', type=str, help='Path to the input reduced basis configuration file.')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
   
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    input_reduced_basis_args = load_yaml(args.input_reduced_basis_config_path)
    KLE_args = input_reduced_basis_args['KLE']
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)
    
    print(f'Running: {sys.argv[0]}')

    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    temp_dolfin_vector = dolfin.Vector(MPI.COMM_SELF, GRF.Vh['parameter'].dim())
    random_test_matrix = hippylib.MultiVector(temp_dolfin_vector, KLE_args['num_reduced_basis']+KLE_args['oversampling'])
    random_generator = hippylib.Random(KLE_args['seed'])  
    random_generator.normal(1., random_test_matrix)

    kle_operator = KLEOperator(GRF.prior)
   
    start_time = MPI.Wtime()
    eigvals, eigvecs = hippylib.doublePassG(A=kle_operator,
                                            B=GRF.prior.M,
                                            Binv=GRF.prior.Msolver,
                                            Omega=random_test_matrix, 
                                            k=KLE_args['num_reduced_basis'], 
                                            s=1, 
                                            check=False)
    end_time = MPI.Wtime()
    print(f'doublePassG elapsed time: {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    kle_basis_nodal_values = numpy.zeros((KLE_args['num_reduced_basis'], GRF.Vh['parameter'].dim()))
    for i in range(KLE_args['num_reduced_basis']):
        kle_basis_nodal_values[i,:] = eigvecs[i].get_local()
    save_npy(input_reduced_basis_path+'/KLE/eigenvalues.npy', eigvals[:KLE_args['num_reduced_basis']])
    save_npy(input_reduced_basis_path+'/KLE/nodal_values.npy', kle_basis_nodal_values) 

    
            