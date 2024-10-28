import os
import sys
import numpy
import argparse

import dolfin

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from data_generation.differential_equations import Hyperelasticity # noqa
from utils import load_yaml, load_npy, save_npy, timing # noqa 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the evaluation labels.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    args = parser.parse_args()
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    print(f'Running: {sys.argv[0]}')
    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
    num_vertices = hyperelasticity.mesh.num_vertices()
    coordinates = hyperelasticity.mesh.coordinates()

    for dataset_path in [args.train_dataset_path, args.test_dataset_path]:
        input_nodal_values = load_npy(dataset_path+'/input_functions/nodal_values.npy')
        output_nodal_values = load_npy(dataset_path+'/output_functions/nodal_values.npy')

        num_functions = input_nodal_values.shape[0]
        input_vertex_values = numpy.zeros((num_functions, num_vertices))
        output_vertex_values = numpy.zeros((num_functions, num_vertices, 2))


        for i in range(num_functions):
            m = dolfin.Function(hyperelasticity.Vh['parameter'])
            m.vector().set_local(input_nodal_values[i,:])
            input_vertex_values[i,:] = m.compute_vertex_values(hyperelasticity.mesh)

        for i in range(num_functions):
            u = dolfin.Function(hyperelasticity.Vh['state'])
            u.vector().set_local(output_nodal_values[i,:])
            u1, u2 = u.split(deepcopy=True)
            output_vertex_values[i,:,0] = u1.compute_vertex_values(hyperelasticity.mesh)
            output_vertex_values[i,:,1] = u2.compute_vertex_values(hyperelasticity.mesh)

        save_npy(dataset_path+'/input_functions/vertex_values.npy', input_vertex_values)
        save_npy(dataset_path+'/input_functions/coordinates.npy', coordinates)
        save_npy(dataset_path+'/output_functions/vertex_values.npy', output_vertex_values)
        save_npy(dataset_path+'/output_functions/coordinates.npy', coordinates)
