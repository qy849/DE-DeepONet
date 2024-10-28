import os
import sys
import numpy
import dolfin
from mpi4py import MPI
import hippylib

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')
from utils import get_mass_matrix, get_stiffness_matrix # noqa

class PODOperator:
    def __init__(self, nodal_values: numpy.ndarray, weighted_matrix: dolfin.Matrix):
        self.nodal_values = nodal_values
        self.weighted_matrix = weighted_matrix

    @property
    def num_functions(self):
        return self.nodal_values.shape[0]

    @property
    def num_nodes(self):
        return self.nodal_values.shape[1]

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):
        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        for i in range(x.nvec()):
            vec_1[:] = self.nodal_values.T @ x[i].get_local()
            self.weighted_matrix.mult(vec_1, vec_2)
            y[i][:] = self.nodal_values @ vec_2.get_local()


class L2PODOperator:
    def __init__(self, nodal_values: numpy.ndarray, Vh: dolfin.FunctionSpace):
        self.nodal_values = nodal_values
        self.Vh = Vh

        self.num_functions, self.num_nodes = self.nodal_values.shape
        self.mass_matrix = get_mass_matrix(self.Vh)

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):
        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        for i in range(x.nvec()):
            vec_1[:] = self.nodal_values.T @ x[i].get_local()
            self.mass_matrix.mult(vec_1, vec_2)
            y[i][:] = self.nodal_values @ vec_2.get_local()



class H1PODOperator:
    def __init__(self, nodal_values: numpy.ndarray, Vh: dolfin.FunctionSpace):
        self.nodal_values = nodal_values
        self.Vh = Vh

        self.num_functions, self.num_nodes = self.nodal_values.shape
        self.mass_matrix = get_mass_matrix(self.Vh)
        self.stiffness_matrix = get_stiffness_matrix(self.Vh)

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):
        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        vec_3 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        for i in range(x.nvec()):
            vec_1[:] = self.nodal_values.T @ x[i].get_local()
            self.mass_matrix.mult(vec_1, vec_2)
            self.stiffness_matrix.mult(vec_1, vec_3)
            y[i][:] = self.nodal_values @ (vec_2.get_local() + vec_3.get_local())