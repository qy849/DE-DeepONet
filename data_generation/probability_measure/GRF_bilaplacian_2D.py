import os
import sys
import argparse

import numpy
import dolfin
import hippylib

from mpi4py import MPI

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)

from utils import load_yaml, save_npy, get_nodal_values, timing, format_elapsed_time, get_split # noqa


class GaussianRandomField:
    def __init__(self, mesh_args, function_space_args, gaussian_random_field_args):
        """
            mesh_args ={
                'length_x': 1.0,
                'length_y': 1.0,
                'num_x': 32,
                'num_y': 32,
            }
        
            function_space_args = {
                'parameter': {
                    'family': 'CG',
                    'degree': 1,
                },
                'state': None
            }

            gaussian_random_field_args = {
                'gamma': 1.0,
                'delta': 1.0,
                'mean': 0.0,
                'robin_bc': True,
                'num_train': 100,
                'num_test': 100,
                'seed': 0,
            }
        """

        self.mesh_args = mesh_args
        self.function_space_args = function_space_args
        self.gaussian_random_field_args = gaussian_random_field_args

        self._mesh = None
        self._Vh = None
        self._prior = None

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = dolfin.RectangleMesh(MPI.COMM_SELF, 
                                             dolfin.Point(0.0, 0.0), 
                                             dolfin.Point(self.mesh_args['length_x'], self.mesh_args['length_y']), 
                                             self.mesh_args['num_x'], 
                                             self.mesh_args['num_y'],
                                             self.mesh_args['diagonal'])
        return self._mesh

    @property 
    def Vh(self):
        if self._Vh is None:
            self._Vh = {
                'parameter': dolfin.FunctionSpace(self.mesh, 
                                                  self.function_space_args['parameter']['family'], 
                                                  self.function_space_args['parameter']['degree']),
                'state': None
            }
        return self._Vh


    @property
    def prior(self):
        if self._prior is None:
            mean_vector = dolfin.Vector(MPI.COMM_SELF, self.Vh['parameter'].dim())
            mean_vector.set_local(numpy.ones(self.Vh['parameter'].dim())*self.gaussian_random_field_args['mean'])
            self._prior = hippylib.BiLaplacianPrior(Vh=self.Vh['parameter'], 
                                                    gamma=self.gaussian_random_field_args['gamma'], 
                                                    delta=self.gaussian_random_field_args['delta'],
                                                    mean=mean_vector,
                                                    robin_bc=self.gaussian_random_field_args['robin_bc'])
        return self._prior


    def sample(self, num_functions: int, seed: int=0) -> list[dolfin.Function]:
        sample_functions = []
        random_generator = hippylib.Random(seed)
        for _ in range(num_functions):
            white_noise_vector = dolfin.Vector()
            self.prior.init_vector(white_noise_vector, 'noise')
            random_generator.normal(1., white_noise_vector)
            sample_vector = dolfin.Vector()
            self.prior.init_vector(sample_vector, 0)
            self.prior.sample(white_noise_vector,sample_vector)
            sample_function = hippylib.vector2Function(sample_vector, self.Vh['parameter'])
            sample_functions.append(sample_function)
        
        return sample_functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 2D Gaussian random fields with Matern covariance function.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')

    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    dolfin.set_log_active(False)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)
    num_functions = gaussian_random_field_args['num_train'] + gaussian_random_field_args['num_test']

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

    local_seed = gaussian_random_field_args['seed'] + rank

    if rank == 0:
        start_time = MPI.Wtime()
    local_sample_functions = GRF.sample(num_functions=split_num_functions[rank], seed=local_seed)
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'GRF sampling elapsed time (rank 0, {split_num_functions[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    local_nodal_values = get_nodal_values(functions=local_sample_functions, dtype='float64')
    
    if rank == 0: 
        nodal_values = numpy.empty([num_functions, GRF.Vh['parameter'].dim()], dtype='float64')
    else:
        nodal_values = None

    comm.Gatherv(local_nodal_values, [nodal_values, counts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        save_npy(train_dataset_path+'/input_functions/nodal_values.npy', nodal_values[:gaussian_random_field_args['num_train'],:])
        save_npy(test_dataset_path+'/input_functions/nodal_values.npy', nodal_values[gaussian_random_field_args['num_train']:,:])