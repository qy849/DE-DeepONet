import numpy
import hippylib
from mpi4py import MPI

class AverageOperator:

    def __init__(self, local_operator, comm):

        if not hasattr(local_operator, 'matMvMult'):
            raise ValueError('local_operator should have method "matMvMult".')

        self.local_operator = local_operator
        self.comm = comm

    def matMvMult(self, x: hippylib.MultiVector, y: hippylib.MultiVector):

        self.local_operator.matMvMult(x,y)
        for i in range(y.nvec()):
            local_array = y[i].get_local()
            all_array = numpy.zeros_like(local_array)
            self.comm.Allreduce(local_array, all_array, op=MPI.SUM)
            y[i][:] = all_array / float(self.comm.Get_size())
