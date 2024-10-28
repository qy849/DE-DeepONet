import dolfin
from mpi4py import MPI
import hippylib

class KLEOperator:

    def __init__(self, GRF: hippylib.SqrtPrecisionPDE_Prior):

        self.GRF = GRF

    @property
    def Vh(self):
        return self.GRF.Vh

    @property
    def M(self):
        return self.GRF.M

    @property
    def Rsolver(self):
        return self.GRF.Rsolver

    @property
    def num_nodes(self):
        return self.Vh.dim()

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):

        vec_1 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        vec_2 = dolfin.Vector(MPI.COMM_SELF, self.num_nodes)
        for i in range(x.nvec()):
            vec_1[:] = x[i].get_local()
            self.M.mult(vec_1, vec_2)
            self.Rsolver.solve(vec_1, vec_2)
            self.M.mult(vec_1,vec_2)
            y[i][:] = vec_2.get_local()