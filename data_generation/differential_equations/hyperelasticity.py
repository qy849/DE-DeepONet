import os
import sys
import argparse

import numpy
import dolfin 
from mpi4py import MPI # should be placed after "import dolfin" to avoid a potential bug 

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_yaml, format_elapsed_time, load_and_scatter, gather_and_save, timing # noqa 

class Hyperelasticity:
    def __init__(self, mesh_args, function_space_args):
        """
            mesh_args ={
                'length_x': 1.0,
                'length_y': 1.0,
                'num_x': 64,
                'num_y': 64,
                'diagonal': 'right'
            }
        
            function_space_args = {
                'parameter': {
                    'family': 'CG',
                    'degree': 1,
                },
                'state': {
                    'family': 'CG',
                    'degree': 1,
                }
            }
        """

        self.mesh_args = mesh_args
        self.function_space_args = function_space_args

        self._mesh = None
        self._Vh = None
        self._bcs = None
        self._bcs0 = None
        self._right_traction_values = None
        self._ds = None
  
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
                'state': dolfin.VectorFunctionSpace(self.mesh, 
                                                   self.function_space_args['state']['family'], 
                                                   self.function_space_args['state']['degree'])
            }
        return self._Vh

    @property
    def bcs(self):
        if self._bcs is None:
            left_boundary_subdomain = dolfin.CompiledSubDomain('near(x[0], side) && on_boundary', side=0.0)
            left_boundary_values = dolfin.Constant((0.0, 0.0))
            self._bcs = []
            self._bcs.append(dolfin.DirichletBC(self.Vh['state'], left_boundary_values, left_boundary_subdomain))
        return self._bcs

    @property
    def bcs0(self):
        if self._bcs0 is None:
            left_boundary_subdomain = dolfin.CompiledSubDomain('near(x[0], side) && on_boundary', side=0.0)
            left_boundary_values = dolfin.Constant((0.0, 0.0))
            self._bcs0 = []
            self._bcs0.append(dolfin.DirichletBC(self.Vh['state'], left_boundary_values, left_boundary_subdomain))
        return self._bcs0

    @property
    def right_traction_values(self):
        if self._right_traction_values is None:
            right_traction_expression = dolfin.Expression(('a*exp(-1.0*pow(x[1]-0.5,2)/b)', 'c*(1.0+(x[1]/d))'), 
                                                            a=0.06, b=4, c=0.03, d=10, degree=5)
            self._right_traction_values = dolfin.interpolate(right_traction_expression, self.Vh['state'])
        return self._right_traction_values


    @property
    def ds(self):
        if self._ds is None:
            right_boundary_subdomain = dolfin.CompiledSubDomain('near(x[0], side) && on_boundary', side=self.mesh_args['length_x'])
            dim_facet = self.mesh.topology().dim() - 1
            boundary_markers = dolfin.MeshFunction('size_t', self.mesh, dim_facet)
            boundary_markers.set_all(0)
            right_boundary_subdomain.mark(boundary_markers, 1)
            self._ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)
        return self._ds

    def compile_form(self, m, u, v):
        E_L = 1.0 
        nu = 0.4 # Poisson ratio
        E = dolfin.exp(m) + dolfin.Constant(E_L) # Young's modulus of elasticity

        Id = dolfin.Identity(u.geometric_dimension()) # identity matrix
        F = Id + dolfin.grad(u) # deformation gradient
        C = dolfin.dot(F.T, F) # right Cauchy–Green strain
        lambda_ = (E*nu)/((1.0+nu)*(1.0-2.0*nu)) # Lame parameters
        mu = E/(2.0*(1.0+nu)) # Lame parameters
        det_F = dolfin.det(F)
        tr_C = dolfin.tr(C)
        W = (mu/2.0) * (tr_C - 3.0) + (lambda_/2.0)*(dolfin.ln(det_F))**2 - mu*dolfin.ln(det_F)

        energy = W*dolfin.dx - dolfin.inner(self.right_traction_values, u) * self.ds(1)
        form = dolfin.derivative(energy, u, v)
        return form

    def solve(self, m):
        u = dolfin.Function(self.Vh['state'])
        v = dolfin.TestFunction(self.Vh['state'])
        form = self.compile_form(m=m, u=u, v=v)
        dolfin.solve(form == 0, u, self.bcs)
        return u


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the hyperelasticity problem.')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file')
    parser.add_argument('--train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, help='Path to the test dataset')
    args = parser.parse_args() 
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    hyperelasticity = Hyperelasticity(mesh_args, function_space_args)
    
    for dataset_path in [train_dataset_path, test_dataset_path]:
        local_input_nodal_values, split_num_functions = load_and_scatter(comm, dataset_path+'/input_functions/nodal_values.npy')
        local_output_nodal_values = numpy.empty((split_num_functions[rank], hyperelasticity.Vh['state'].dim()), dtype='float64')
        if rank == 0:
            start_time = MPI.Wtime()
        for i in range(split_num_functions[rank]):
            m = dolfin.Function(hyperelasticity.Vh['parameter'])
            m.vector().set_local(local_input_nodal_values[i,:])
            u = hyperelasticity.solve(m=m)
            local_output_nodal_values[i,:] = u.vector().get_local()
        if rank == 0:
            end_time = MPI.Wtime()
            print(f'PDE solves elapsed time (rank 0, {split_num_functions[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')
        gather_and_save(comm, dataset_path+'/output_functions/nodal_values.npy', local_output_nodal_values, split_num_functions)





