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
from data_generation.differential_equations import NavierStokes # noqa
from data_generation.input_reduced_basis import ModifiedInputActiveOperator # noqa
from utils import load_yaml, load_npy, save_npy, load_and_scatter, gather, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the derivative m labels for testing Jacobian action.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    parser.add_argument('--test_dm_path', type=str, help='Path to the test_dm directory')
   
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    test_dataset_path = args.test_dataset_path
    test_dm_path = args.test_dm_path

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    navier_stokes = NavierStokes(mesh_args, function_space_args)
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    input_dim = navier_stokes.Vh['parameter'].dim()
    output_dim = navier_stokes.Vh['state'].dim()

    dm_direction_nodal_values = load_npy(test_dm_path+'/dm_direction_nodal_values.npy')
    local_input_nodal_values, split = load_and_scatter(comm, test_dataset_path+'/input_functions/nodal_values.npy')
    local_output_nodal_values, split = load_and_scatter(comm, test_dataset_path+'/output_functions/nodal_values.npy')

    input_functions = []
    output_functions = []
    for i in range(split[rank]):
        input_function = dolfin.Function(navier_stokes.Vh['parameter'])
        input_function.vector().set_local(local_input_nodal_values[i,:])
        input_functions.append(input_function)
        output_function = dolfin.Function(navier_stokes.Vh['state'])
        output_function.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(output_function)


    num_directions = dm_direction_nodal_values.shape[0]
    dm_direction_functions = []
    for i in range(num_directions):
        dm_direction_function = dolfin.Function(navier_stokes.Vh['parameter'])
        dm_direction_function.vector().set_local(dm_direction_nodal_values[i,:])
        dm_direction_functions.append(dm_direction_function)


    local_derivative_labels = numpy.zeros((split[rank], navier_stokes.mesh.num_vertices(), num_directions, 2))


    input_active_operator = ModifiedInputActiveOperator(m_list=input_functions,
                                                        u_list=output_functions,
                                                        compile_form=navier_stokes.compile_form,
                                                        bcs0=navier_stokes.bcs0,
                                                        prior=GRF.prior)

    for i in range(split[rank]):
        derivative_functions = input_active_operator.jacobian_action(input_functions[i], output_functions[i], dm_direction_functions)
        for j in range(num_directions):
            dm_velocity, _ = derivative_functions[j].split(deepcopy=True)
            dm_1, dm_2 = dm_velocity.split(deepcopy=True)
            local_derivative_labels[i,:,j,0] = dm_1.compute_vertex_values(navier_stokes.mesh)
            local_derivative_labels[i,:,j,1] = dm_2.compute_vertex_values(navier_stokes.mesh)

   
    coordinates = navier_stokes.mesh.coordinates()
    sub_mesh = dolfin.RectangleMesh(MPI.COMM_SELF,
                                    dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], 
                                    mesh_args['num_y'],
                                    'right')
    sub_coordinates = sub_mesh.coordinates()
    matching_indices = [numpy.where(column)[0][0] for column in (coordinates[:, None, :] == sub_coordinates).all(axis=-1).T]
    local_derivative_labels = local_derivative_labels[:, matching_indices, :, :]
    local_samples_start_index = sum(split[:rank])
    local_samples_end_index = local_samples_start_index + split[rank]
    save_npy(test_dm_path+f'/derivative_m_labels_{local_samples_start_index}_{local_samples_end_index}.npy', local_derivative_labels)