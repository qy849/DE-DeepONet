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

from utils import load_yaml, save_npy, get_nodal_values, format_elapsed_time, get_split, timing # noqa
from data_generation.probability_measure import GaussianRandomField # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the random basis of input function space.')
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
    Random_args = input_reduced_basis_args['Random']
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)
    num_functions = Random_args['num_reduced_basis']

    if rank == 0:
        split_num_functions = get_split(N=num_functions, size=size)
        counts = [num * GRF.Vh['parameter'].dim() for num in split_num_functions]
        displacements = [sum(counts[:i]) for i in range(size)]
    else:
        split_num_functions = None
        counts = None
        displacements = None
        
    split_num_functions = comm.bcast(split_num_functions, root=0)
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    local_seed = Random_args['seed'] + rank
    local_sample_functions = GRF.sample(num_functions=split_num_functions[rank], seed=local_seed)
    local_nodal_values = get_nodal_values(functions=local_sample_functions, dtype='float64')
    
    if rank == 0: 
        nodal_values = numpy.empty([num_functions, GRF.Vh['parameter'].dim()], dtype='float64')
    else:
        nodal_values = None

    comm.Gatherv(local_nodal_values, [nodal_values, counts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        save_npy(input_reduced_basis_path+'/Random/nodal_values.npy', nodal_values)





