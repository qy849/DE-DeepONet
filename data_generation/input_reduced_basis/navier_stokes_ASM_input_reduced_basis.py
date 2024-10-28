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

from data_generation.probability_measure import GaussianRandomField # noqa
from data_generation.differential_equations import NavierStokes # noqa
from data_generation.operators import InputActiveOperator, AverageOperator # noqa
from utils import load_yaml, save_npy, load_and_scatter, get_mass_matrix, timing, format_elapsed_time # noqa


class ModifiedInputActiveOperator(InputActiveOperator):
    def __init__(self, 
                 m_list: list[dolfin.Function], 
                 u_list: list[dolfin.Function], 
                 compile_form: callable,
                 bcs0: list[dolfin.DirichletBC],
                 prior: hippylib.SqrtPrecisionPDE_Prior):
        super().__init__(m_list=m_list, u_list=u_list, compile_form=compile_form, bcs0=bcs0, prior=prior)

        self.Vh = {
            'parameter': self.m_list[0].function_space(),
            'state': self.u_list[0].function_space()
        }
        self.Vh['velocity'] = self.Vh['state'].sub(0).collapse()
        self.Vh['pressure'] = self.Vh['state'].sub(1).collapse()

        self.M = {
            'parameter': get_mass_matrix(self.Vh['parameter']),
            'state': get_mass_matrix(self.Vh['state']),
            'velocity': get_mass_matrix(self.Vh['velocity']),
            'pressure': get_mass_matrix(self.Vh['pressure'])
        }

        self.Msolver = {    
            'parameter': dolfin.LUSolver(self.M['parameter'], 'mumps'),
            'state': dolfin.LUSolver(self.M['state'], 'mumps'),
            'velocity': dolfin.LUSolver(self.M['velocity'], 'mumps'),
            'pressure': dolfin.LUSolver(self.M['pressure'], 'mumps')
        }

    def action(self, m: dolfin.Function, u: dolfin.Function, input_functions: list[dolfin.Function]) -> list[dolfin.Vector]:
        temp_functions = super().jacobian_action(m, u, input_functions)
        for _, temp_function in enumerate(temp_functions):
            velocity, pressure = temp_function.split(deepcopy=True)
            pressure.vector().zero()
            dolfin.assign(temp_function.sub(1), pressure)

        return super().adjoint_jacobian_action(m, u, temp_functions)

    def compute_reduced_jacobian(self, 
                                input_reduced_basis_nodal_values: numpy.ndarray, 
                                output_reduced_basis_nodal_values: numpy.ndarray) -> numpy.ndarray:

        num_input_reduced_basis = input_reduced_basis_nodal_values.shape[0]
        num_output_reduced_basis = output_reduced_basis_nodal_values.shape[0]

        input_reduced_basis_functions = []
        for i in range(num_input_reduced_basis):
            input_reduced_basis_function = dolfin.Function(self.Vh['parameter'])
            input_reduced_basis_function.vector().set_local(input_reduced_basis_nodal_values[i,:])
            input_reduced_basis_functions.append(input_reduced_basis_function)    
        
        mtx_1 = numpy.zeros((self.num_functions, num_input_reduced_basis, self.Vh['velocity'].dim()))
        for i in range(self.num_functions):
            m = self.m_list[i]
            u = self.u_list[i]
            dm_functions = self.jacobian_action(m=m, u=u, input_functions=input_reduced_basis_functions)
            for j,dm_function in enumerate(dm_functions):
                dm_velocity, dm_pressure = dm_function.split(deepcopy=True)
                mtx_1[i,j,:] = dm_velocity.vector().get_local()
            
        mtx_2 = numpy.zeros((num_output_reduced_basis, self.Vh['velocity'].dim()))
        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.Vh['velocity'].dim())
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.Vh['velocity'].dim())

        for i in range(num_output_reduced_basis):
            vec_1[:] = output_reduced_basis_nodal_values[i,:]
            self.M['velocity'].mult(vec_1, vec_2)
            mtx_2[i,:] = vec_2.get_local()

        reduced_jacobian = numpy.einsum('ijk, lk->ilj', mtx_1, mtx_2)
        return reduced_jacobian


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the active subspace basis of input function space.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file.')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file.')
    parser.add_argument('--gaussian_random_field_config_path', type=str, help='Path to the Gaussian random field configuration file.')
    parser.add_argument('--input_reduced_basis_config_path', type=str, help='Path to the input reduced basis configuration file.')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis.')
   
    args = parser.parse_args()
    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    gaussian_random_field_args = load_yaml(args.gaussian_random_field_config_path)
    input_reduced_basis_args = load_yaml(args.input_reduced_basis_config_path)
    ASM_args = input_reduced_basis_args['ASM']
    train_dataset_path = args.train_dataset_path
    input_reduced_basis_path = args.input_reduced_basis_path

    dolfin.set_log_active(False)

    navier_stokes = NavierStokes(mesh_args, function_space_args)
    GRF = GaussianRandomField(mesh_args, function_space_args, gaussian_random_field_args)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    local_input_nodal_values, split = load_and_scatter(comm, train_dataset_path+'/input_functions/nodal_values.npy', start_index=0, end_index=ASM_args['num_grad'])
    local_output_nodal_values, split = load_and_scatter(comm, train_dataset_path+'/output_functions/nodal_values.npy', start_index=0, end_index=ASM_args['num_grad'])

    random_test_matrix_shape = (ASM_args['num_reduced_basis'] + ASM_args['oversampling'], navier_stokes.Vh['parameter'].dim())
    if rank == 0: 
        numpy.random.seed(ASM_args['seed'])
        temp_matrix = numpy.random.randn(*random_test_matrix_shape)
    else:
        temp_matrix = None

    temp_matrix = comm.bcast(temp_matrix, root=0)

    temp_dolfin_vector = dolfin.Vector(MPI.COMM_SELF, navier_stokes.Vh['parameter'].dim())
    random_test_matrix = hippylib.MultiVector(temp_dolfin_vector, random_test_matrix_shape[0])
        
    for i in range(random_test_matrix_shape[0]):
        random_test_matrix[i][:] = temp_matrix[i,:]

    input_functions = []
    output_functions = []
    for i in range(split[rank]):
        input_function = dolfin.Function(navier_stokes.Vh['parameter'])
        input_function.vector().set_local(local_input_nodal_values[i,:])
        input_functions.append(input_function)
        output_function = dolfin.Function(navier_stokes.Vh['state'])
        output_function.vector().set_local(local_output_nodal_values[i,:])
        output_functions.append(output_function)

    input_active_operator = ModifiedInputActiveOperator(m_list=input_functions,
                                                        u_list=output_functions,
                                                        compile_form=navier_stokes.compile_form,
                                                        bcs0=navier_stokes.bcs0,
                                                        prior=GRF.prior)
    average_input_active_operator = AverageOperator(local_operator=input_active_operator, comm=comm)
    precision_operator = GRF.prior.R
    covariance_solver = GRF.prior.Rsolver

    if rank == 0:
        start_time = MPI.Wtime()
    eigvals, eigvecs = hippylib.doublePassG(A=average_input_active_operator, 
                                            B=precision_operator, 
                                            Binv=covariance_solver, 
                                            Omega=random_test_matrix, 
                                            k=ASM_args['num_reduced_basis'], 
                                            s=1, 
                                            check=False)
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'doublePassG elapsed time: {format_elapsed_time(start_time=start_time, end_time=end_time)}')

    if rank == 0: 
        active_basis_nodal_values = numpy.zeros((ASM_args['num_reduced_basis'], navier_stokes.Vh['parameter'].dim()))
        for i in range(ASM_args['num_reduced_basis']):
            active_basis_nodal_values[i,:] = eigvecs[i].get_local()
        save_npy(input_reduced_basis_path+'/ASM/eigenvalues.npy', eigvals[:ASM_args['num_reduced_basis']])
        save_npy(input_reduced_basis_path+'/ASM/nodal_values.npy', active_basis_nodal_values)


    




