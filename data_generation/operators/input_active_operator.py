import os
import sys
import numpy
import dolfin
from mpi4py import MPI
import hippylib

from .jacobian_operator import JacobianOperator

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')
from utils import get_mass_matrix # noqa
class InputActiveOperator(JacobianOperator):

    def __init__(self, 
                 m_list: list[dolfin.Function], 
                 u_list: list[dolfin.Function], 
                 compile_form: callable,
                 bcs0: list[dolfin.DirichletBC],
                 prior: hippylib.SqrtPrecisionPDE_Prior):

        self.m_list = m_list
        self.u_list = u_list
        super().__init__(compile_form=compile_form, bcs0=bcs0)
        self.prior = prior

        assert len(m_list) == len(u_list)
        self.num_functions = len(self.m_list)
        self.Vh = {
            'parameter': self.m_list[0].function_space(),
            'state': self.u_list[0].function_space()
        }
        self.M = {
            'parameter': get_mass_matrix(self.Vh['parameter']),
            'state': get_mass_matrix(self.Vh['state'])
        }

        self.Msolver = {    
            'parameter': dolfin.LUSolver(self.M['parameter'], 'mumps'),
            'state': dolfin.LUSolver(self.M['state'], 'mumps')
        }

    def action(self, m: dolfin.Function, u: dolfin.Function, input_functions: list[dolfin.Function]) -> list[dolfin.Vector]:
        return super().adjoint_jacobian_action(m, u, super().jacobian_action(m, u, input_functions))

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):
        num_input_functions = x.nvec()
        input_functions = []
        
        for i in range(num_input_functions):
            input_function = dolfin.Function(self.Vh['parameter'])
            input_function.vector().set_local(x[i])
            input_functions.append(input_function)        
        
        mean_output_function_nodal_values = numpy.zeros((num_input_functions, self.Vh['parameter'].dim()))
        for i in range(self.num_functions):
            m = self.m_list[i]
            u = self.u_list[i]
            output_functions_projection = self.action(m=m, u=u, input_functions=input_functions)
            for j, projection in enumerate(output_functions_projection):
                mean_output_function_nodal_values[j,:] += projection.get_local()
        
        mean_output_function_nodal_values /= self.num_functions
        for i in range(num_input_functions):
            y[i][:] = mean_output_function_nodal_values[i,:]


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
        
        mtx_1 = numpy.zeros((self.num_functions, num_input_reduced_basis, self.Vh['state'].dim()))
        for i in range(self.num_functions):
            m = self.m_list[i]
            u = self.u_list[i]
            dm_functions = self.jacobian_action(m=m, u=u, input_functions=input_reduced_basis_functions)
            for j,dm_function in enumerate(dm_functions):
                mtx_1[i,j,:] = dm_function.vector().get_local()
            
        mtx_2 = numpy.zeros((num_output_reduced_basis, self.Vh['state'].dim()))
        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.Vh['state'].dim())
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.Vh['state'].dim())

        for i in range(num_output_reduced_basis):
            vec_1[:] = output_reduced_basis_nodal_values[i,:]
            self.M['state'].mult(vec_1, vec_2)
            mtx_2[i,:] = vec_2.get_local()

        reduced_jacobian = numpy.einsum('ijk, lk->ilj', mtx_1, mtx_2)
        return reduced_jacobian