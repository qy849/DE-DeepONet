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

from data_generation.differential_equations import NavierStokes # noqa
from data_generation.operators import L2PODOperator # noqa
from utils import load_yaml, load_npy, save_npy, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate POD basis of output function space.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--output_reduced_basis_config_path', type=str, help='Path to the output reduced basis configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--output_reduced_basis_path', type=str, help='Path to the output reduced basis.')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    output_reduced_basis_args = load_yaml(args.output_reduced_basis_config_path)
    POD_args = output_reduced_basis_args['POD']
    train_dataset_path = args.train_dataset_path
    output_reduced_basis_path = args.output_reduced_basis_path

    dolfin.set_log_active(False)
    print(f'Running: {sys.argv[0]}')

    output_nodal_values = load_npy(train_dataset_path+'/output_functions/nodal_values.npy')
    if POD_args['num_eval'] > output_nodal_values.shape[0]:
        num_eval = output_nodal_values.shape[0]
        print(f'Only {output_nodal_values.shape[0]} output functions available.')
    
    selected_output_nodal_values = output_nodal_values[:POD_args['num_eval'],:]
    if POD_args['num_eval'] < POD_args['num_reduced_basis']:
        raise ValueError(f'num_eval {POD_args["num_eval"]} should be larger than or equal to num_reduced_basis {POD_args["num_reduced_basis"]}')


    navier_stokes = NavierStokes(mesh_args, function_space_args)

    selected_velocity_output_nodal_values = numpy.zeros((selected_output_nodal_values.shape[0], navier_stokes.Vh['velocity'].dim()))
    for i in range(selected_output_nodal_values.shape[0]):
        up = dolfin.Function(navier_stokes.Vh['state'])
        up.vector().set_local(selected_output_nodal_values[i,:])
        u, p = up.split(deepcopy=True)
        selected_velocity_output_nodal_values[i,:] = u.vector().get_local()

    temp_dolfin_vector = dolfin.Vector(MPI.COMM_SELF, POD_args['num_eval'])
    random_test_matrix = hippylib.MultiVector(temp_dolfin_vector, POD_args['num_reduced_basis']+POD_args['oversampling'])
    random_generator = hippylib.Random(POD_args['seed'])  
    random_generator.normal(1., random_test_matrix)

    pod_operator = L2PODOperator(nodal_values=selected_velocity_output_nodal_values, Vh=navier_stokes.Vh['velocity'])

    eigvals, eigvecs = hippylib.doublePass(A=pod_operator, 
                                           Omega=random_test_matrix, 
                                           k=POD_args['num_reduced_basis'], 
                                           s=1, 
                                           check=False)

    pod_basis_nodal_values = numpy.zeros((POD_args['num_reduced_basis'], navier_stokes.Vh['velocity'].dim()))
    for i in range(POD_args['num_reduced_basis']):
        pod_basis_nodal_values[i,:] = selected_velocity_output_nodal_values.T @ eigvecs[i].get_local() / numpy.sqrt(eigvals[i])
    save_npy(output_reduced_basis_path+'/POD/eigenvalues.npy', eigvals[:POD_args['num_reduced_basis']])
    save_npy(output_reduced_basis_path+'/POD/nodal_values.npy', pod_basis_nodal_values)