import os
import sys
import numpy
import argparse

import dolfin

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import NavierStokes # noqa
from utils import load_yaml, load_npy, save_npy, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the evaluation labels.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    dataset_path = args.dataset_path

    print(f'Running: {sys.argv[0]}')
    navier_stokes = NavierStokes(mesh_args, function_space_args)
    num_vertices = navier_stokes.mesh.num_vertices()
    coordinates = navier_stokes.mesh.coordinates()

    sub_mesh = dolfin.RectangleMesh(dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], 
                                    mesh_args['num_y'],
                                    'right')
    sub_coordinates = sub_mesh.coordinates()
    matching_indices = [numpy.where(column)[0][0] for column in (coordinates[:, None, :] == sub_coordinates).all(axis=-1).T]

    input_nodal_values = load_npy(dataset_path+'/input_functions/nodal_values.npy')
    output_nodal_values = load_npy(dataset_path+'/output_functions/nodal_values.npy')

    num_functions = input_nodal_values.shape[0]
    input_vertex_values = numpy.zeros((num_functions, num_vertices))
    output_vertex_values = {
        'velocity': numpy.zeros((num_functions, num_vertices, 2)),
        'pressure': numpy.zeros((num_functions, num_vertices))
    }

    velocity_nodal_values = numpy.zeros((num_functions, navier_stokes.Vh['velocity'].dim()), dtype='float64')
    pressure_nodal_values = numpy.zeros((num_functions, navier_stokes.Vh['pressure'].dim()), dtype='float64')

    for i in range(num_functions):
        m = dolfin.Function(navier_stokes.Vh['parameter'])
        m.vector().set_local(input_nodal_values[i,:])
        input_vertex_values[i,:] = m.compute_vertex_values(navier_stokes.mesh)

    for i in range(num_functions):
        u = dolfin.Function(navier_stokes.Vh['state'])
        u.vector().set_local(output_nodal_values[i,:])
        velocity, pressure = u.split(deepcopy=True)
        velocity_1, velocity_2 = velocity.split(deepcopy=True)
        output_vertex_values['velocity'][i,:,0] = velocity_1.compute_vertex_values(navier_stokes.mesh)
        output_vertex_values['velocity'][i,:,1] = velocity_2.compute_vertex_values(navier_stokes.mesh)
        output_vertex_values['pressure'][i,:] = pressure.compute_vertex_values(navier_stokes.mesh)
        velocity_nodal_values[i,:] = velocity.vector().get_local()
        pressure_nodal_values[i,:] = pressure.vector().get_local()

    save_npy(dataset_path+'/input_functions/vertex_values.npy', input_vertex_values[:,matching_indices])
    save_npy(dataset_path+'/input_functions/coordinates.npy', coordinates)
    save_npy(dataset_path+'/output_functions/vertex_values.npy', output_vertex_values['velocity'][:, matching_indices, :])
    save_npy(dataset_path+'/output_functions/coordinates.npy', coordinates[matching_indices, :])

    save_npy(dataset_path+'/output_functions/velocity_nodal_values.npy', velocity_nodal_values)
    save_npy(dataset_path+'/output_functions/pressure_nodal_values.npy', pressure_nodal_values)