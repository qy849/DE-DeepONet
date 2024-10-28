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

class NavierStokes:
    def __init__(self, mesh_args, function_space_args):
        """
            mesh_args ={
                'length_x': 1.0,
                'length_y': 1.0,
                'num_x': 64,
                'num_y': 64,
                'diagonal': 'crossed'
            }
        
            function_space_args = {
                'parameter': {
                    'family': 'CG',
                    'degree': 1,
                },
                'state': {
                    'u':{
                        'family': 'CG',
                        'degree': 2,
                    }
                    'p':{
                        'family': 'CG',
                        'degree': 1,
                    }
                }
            }
        """

        self.mesh_args = mesh_args
        self.function_space_args = function_space_args

        self._mesh = None
        self._Vh = None
        self._bcs = None
        self._bcs0 = None
  
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
            U_element = dolfin.VectorElement(self.function_space_args['state']['u']['family'], 
                                             self.mesh.ufl_cell(), 
                                             self.function_space_args['state']['u']['degree'])
            P_element = dolfin.FiniteElement(self.function_space_args['state']['p']['family'], 
                                             self.mesh.ufl_cell(), 
                                             self.function_space_args['state']['p']['degree'])
            self._Vh = {
                'parameter': dolfin.FunctionSpace(self.mesh, 
                                                  self.function_space_args['parameter']['family'], 
                                                  self.function_space_args['parameter']['degree']),
                'state': dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([U_element, P_element]))
            }

            self._Vh['velocity'] = self._Vh['state'].sub(0).collapse()
            self._Vh['pressure'] = self._Vh['state'].sub(1).collapse()

        return self._Vh

    @property
    def bcs(self):
        if self._bcs is None:
            top_boundary_subdomain = dolfin.CompiledSubDomain('near(x[1], side) && on_boundary', side=self.mesh_args['length_y'])
            non_top_boundary_subdomain = dolfin.CompiledSubDomain('!near(x[1], side) && on_boundary ', side=self.mesh_args['length_y'])
            pin_point_subdomain = dolfin.CompiledSubDomain('near(x[0], side) && near(x[1], side) && on_boundary', side=0.0)
            top_boundary_values = dolfin.Constant((1.0, 0.0))
            # top_boundary_values = dolfin.Expression(('x[0]*(1.0 - x[0])', '0.0'), degree=2)
            non_top_boundary_values = dolfin.Constant((0.0, 0.0))
            pin_point_value = dolfin.Constant(0.0)
            self._bcs = []
            self._bcs.append(dolfin.DirichletBC(self.Vh['state'].sub(0), top_boundary_values, top_boundary_subdomain))
            self._bcs.append(dolfin.DirichletBC(self.Vh['state'].sub(0), non_top_boundary_values, non_top_boundary_subdomain))
            self._bcs.append(dolfin.DirichletBC(self.Vh['state'].sub(1), pin_point_value, pin_point_subdomain))
        return self._bcs

    @property
    def bcs0(self):
        if self._bcs0 is None:
            top_boundary_subdomain = dolfin.CompiledSubDomain('near(x[1], side) && on_boundary', side=self.mesh_args['length_y'])
            non_top_boundary_subdomain = dolfin.CompiledSubDomain('!near(x[1], side) && on_boundary ', side=self.mesh_args['length_y'])
            pin_point_subdomain = dolfin.CompiledSubDomain('near(x[0], side) && near(x[1], side) && on_boundary', side=0.0)
            top_boundary_values = dolfin.Constant((0.0, 0.0))
            non_top_boundary_values = dolfin.Constant((0.0, 0.0))
            pin_point_value = dolfin.Constant(0.0)
            self._bcs0 = []
            self._bcs0.append(dolfin.DirichletBC(self.Vh['state'].sub(0), top_boundary_values, top_boundary_subdomain))
            self._bcs0.append(dolfin.DirichletBC(self.Vh['state'].sub(0), non_top_boundary_values, non_top_boundary_subdomain))
            self._bcs0.append(dolfin.DirichletBC(self.Vh['state'].sub(1), pin_point_value, pin_point_subdomain))
        return self._bcs0


    def compile_form(self, m, up, vq):
        f = dolfin.Constant((0.0, 0.0))
        u, p = dolfin.split(up)
        v, q = dolfin.split(vq)
        nu = dolfin.exp(m)
        form = (
            dolfin.inner(nu * dolfin.grad(u), dolfin.grad(v))*dolfin.dx 
            + dolfin.inner(dolfin.grad(u)*u, v)*dolfin.dx 
            - dolfin.inner(p, dolfin.div(v))*dolfin.dx 
            + dolfin.inner(f, v)*dolfin.dx 
            + dolfin.inner(q, dolfin.div(u))*dolfin.dx
        )
        return form

    def solve(self, m):
        up = dolfin.Function(self.Vh['state'])
        vq = dolfin.TestFunction(self.Vh['state'])
        form = self.compile_form(m=m, up=up, vq=vq)
        dolfin.solve(form == 0, up, self.bcs)
        return up


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the Navier--Stokes problem')
    parser.add_argument('--mesh_config_path', type=str, help='Path to the mesh configuration file')
    parser.add_argument('--function_space_config_path', type=str, help='Path to the function space configuration file')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args() 
    dolfin.set_log_active(False)

    mesh_args = load_yaml(args.mesh_config_path)
    function_space_args = load_yaml(args.function_space_config_path)
    dataset_path = args.dataset_path

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Running: {sys.argv[0]} with {size} processors')

    navier_stokes = NavierStokes(mesh_args, function_space_args)
    local_input_nodal_values, split_num_functions = load_and_scatter(comm, dataset_path+'/input_functions/nodal_values.npy')
    local_output_nodal_values = numpy.empty((split_num_functions[rank], navier_stokes.Vh['state'].dim()), dtype='float64')
    if rank == 0: 
        start_time = MPI.Wtime()
    for i in range(split_num_functions[rank]):
        m = dolfin.Function(navier_stokes.Vh['parameter'])
        m.vector().set_local(local_input_nodal_values[i,:])
        up = navier_stokes.solve(m=m)
        local_output_nodal_values[i,:] = up.vector().get_local()
    if rank == 0:
        end_time = MPI.Wtime()
        print(f'PDE solves elapsed time (rank 0, {split_num_functions[rank]} samples): {format_elapsed_time(start_time=start_time, end_time=end_time)}')
    gather_and_save(comm, dataset_path+'/output_functions/nodal_values.npy', local_output_nodal_values, split_num_functions)





