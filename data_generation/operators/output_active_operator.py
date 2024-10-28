import os
import sys
import numpy
import dolfin
import hippylib
from mpi4py import MPI 
from .jacobian_operator import JacobianOperator

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')
from utils import get_mass_matrix # noqa

class OutputActiveOperator(JacobianOperator):

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


    def covariance_action(self, input_functions_projection: list[dolfin.Vector]) -> list[dolfin.Function]:
        output_functions = []
        x = dolfin.Vector(MPI.COMM_SELF, self.Vh['parameter'].dim())
        b = dolfin.Vector(MPI.COMM_SELF, self.Vh['parameter'].dim())
        for _,input_function_projection in enumerate(input_functions_projection):
            b[:] = input_function_projection.get_local()
            self.prior.Rsolver.solve(x,b)
            output_function = dolfin.Function(self.Vh['parameter'])
            output_function.vector().set_local(x.get_local())
            output_functions.append(output_function)

        return output_functions


    def action(self, m: dolfin.Function, u: dolfin.Function, input_functions: list[dolfin.Function]) -> list[dolfin.Function]:
        return super().jacobian_action(m, u, self.covariance_action(super().adjoint_jacobian_action(m, u, input_functions)))

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):
        num_input_functions = x.nvec()
        input_functions = []
        
        for i in range(num_input_functions):
            input_function = dolfin.Function(self.Vh['state'])
            input_function.vector().set_local(x[i])
            input_functions.append(input_function)        
        
        mean_output_function_nodal_values = numpy.zeros((num_input_functions, self.Vh['state'].dim()))
        output_function_projection = dolfin.Vector(MPI.COMM_SELF, self.Vh['state'].dim())
        for i in range(self.num_functions):
            m = self.m_list[i]
            u = self.u_list[i]
            output_functions = self.action(m=m, u=u, input_functions=input_functions)
            for j, output_function in enumerate(output_functions):
                self.M['state'].mult(output_function.vector(), output_function_projection)
                mean_output_function_nodal_values[j,:] += output_function_projection.get_local()
        
        mean_output_function_nodal_values /= self.num_functions
        for i in range(num_input_functions):
            y[i][:] = mean_output_function_nodal_values[i,:]
