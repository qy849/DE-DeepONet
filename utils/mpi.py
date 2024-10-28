import numpy
from mpi4py import MPI
from math import prod

def get_split(N: int, size: int) -> list:
    quotient, remainder = divmod(N, size)
    split = [quotient+1 if i < remainder else quotient for i in range(size)]
    return split


def load_and_scatter(comm, dataset_path, start_index=None, end_index=None):
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        array = numpy.load(dataset_path)
        if start_index is None:
            start_index = 0
        else:
            start_index = max(start_index, 0)
        if end_index is None:
            end_index = array.shape[0]
        else:
            end_index = min(end_index, array.shape[0])
        assert start_index < end_index
        N = end_index - start_index
        array = array[start_index:end_index]
        dim = array.shape[1:]
        split = get_split(N, size)        
        counts = [num * prod(dim) for num in split]
        displacements = [sum(counts[:i]) for i in range(size)]
    else:
        array = None
        dim = None
        split = None
        counts = None
        displacements = None

    dim = comm.bcast(dim, root=0)
    split = comm.bcast(split, root=0)
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    local_array = numpy.empty((split[rank], *dim), dtype='float64')
    comm.Scatterv([array, counts, displacements, MPI.DOUBLE], local_array)

    return local_array, split


def gather(comm, local_array, split):
    size = comm.Get_size()
    rank = comm.Get_rank()
    dim = local_array.shape[1:]
    if rank == 0:
        N = sum(split)
        array = numpy.empty((N, *dim), dtype='float64')
        counts = [num * prod(dim) for num in split]
        displacements = [sum(counts[:i]) for i in range(size)]
    else:
        array = None
        counts = None
        displacements = None

    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    comm.Gatherv(local_array, [array, counts, displacements, MPI.DOUBLE], root=0)

    return array


def gather_and_save(comm, dataset_path, local_array, split):
    size = comm.Get_size()
    rank = comm.Get_rank()
    dim = local_array.shape[1:]
    if rank == 0:
        N = sum(split)
        array = numpy.empty((N, *dim), dtype='float64')
        counts = [num * prod(dim) for num in split]
        displacements = [sum(counts[:i]) for i in range(size)]
    else:
        array = None
        counts = None
        displacements = None

    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    comm.Gatherv(local_array, [array, counts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        numpy.save(dataset_path, array)