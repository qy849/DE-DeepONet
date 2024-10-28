import os
import sys
import numpy
import argparse

import dolfin

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, load_npy, save_npy, timing 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the dm direction vertex values.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--dm_direction_nodal_values_path', type=str, help='Path to the dm direction nodal values')
    parser.add_argument('--dm_direction_vertex_values_path', type=str, help='Path to the dm direction vertex values')

    args = parser.parse_args()
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    dm_direction_nodal_values_path = args.dm_direction_nodal_values_path
    dm_direction_vertex_values_path = args.dm_direction_vertex_values_path

    print(f'Running: {sys.argv[0]}')

    mesh = dolfin.RectangleMesh(dolfin.Point(0.0, 0.0), 
                                dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                mesh_args['num_x'], 
                                mesh_args['num_y'],
                                mesh_args['diagonal'])
    Vh = dolfin.FunctionSpace(mesh, function_space_args['parameter']['family'], function_space_args['parameter']['degree'])


    sub_mesh = dolfin.RectangleMesh(dolfin.Point(0.0, 0.0), 
                                    dolfin.Point(mesh_args['length_x'], mesh_args['length_y']), 
                                    mesh_args['num_x'], 
                                    mesh_args['num_y'],
                                    'right')

    coordinates = mesh.coordinates()
    sub_coordinates = sub_mesh.coordinates()
    matching_indices = [numpy.where(column)[0][0] for column in (coordinates[:, None, :] == sub_coordinates).all(axis=-1).T]

    dm_direction_nodal_values = load_npy(dm_direction_nodal_values_path)
    num_dm_direction = dm_direction_nodal_values.shape[0]
    dm_direction_vertex_values = numpy.zeros((dm_direction_nodal_values.shape[0], len(sub_coordinates)))

    for i in range(num_dm_direction):
        u = dolfin.Function(Vh)
        u.vector().set_local(dm_direction_nodal_values[i])
        dm_direction_vertex_values[i,:] = u.compute_vertex_values(mesh)[matching_indices]

    save_npy(dm_direction_vertex_values_path, dm_direction_vertex_values)