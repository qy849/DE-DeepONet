import numpy
import dolfin
from mpi4py import MPI
import hippylib

def get_nodal_values(functions: list[dolfin.Function], dtype: str = 'float64') -> numpy.ndarray:
    function_space = functions[0].function_space()
    num_functions = len(functions)
    num_nodes = function_space.dim()
    nodal_values = numpy.zeros((num_functions, num_nodes), dtype=dtype)
    for i, function in enumerate(functions):
        nodal_values[i, :] = function.vector().get_local()
    return nodal_values

def get_mass_matrix(Vh: dolfin.FunctionSpace) -> dolfin.Matrix:
    u = dolfin.TrialFunction(Vh)
    v = dolfin.TestFunction(Vh)
    mass_matrix = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)
    return mass_matrix

def get_stiffness_matrix(Vh: dolfin.FunctionSpace) -> dolfin.Matrix:
    u = dolfin.TrialFunction(Vh)
    v = dolfin.TestFunction(Vh)
    stiffness_matrix = dolfin.assemble(dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dx)
    return stiffness_matrix

def generate_dof_to_vertex_map(Vh: dolfin.FunctionSpace) -> numpy.ndarray:
    return dolfin.dof_to_vertex_map(Vh)

def generate_vertex_to_dof_map(Vh: dolfin.FunctionSpace) -> numpy.ndarray:
    return dolfin.vertex_to_dof_map(Vh)

def vector2function(nodal_values: numpy.ndarray, Vh: dolfin.FunctionSpace) -> dolfin.Function:
    u = dolfin.Function(Vh)
    u.vector().set_local(nodal_values)
    return u

def M_inner_product(x: dolfin.Vector, y: dolfin.Vector, M: dolfin.Matrix) -> float:
    temp_vector = dolfin.Vector(MPI.COMM_SELF, M.size(0))
    M.mult(y, temp_vector)
    return numpy.dot(x.get_local(), temp_vector.get_local())

def M_norm(x: dolfin.Vector, M: dolfin.Matrix) -> float:
    return numpy.sqrt(M_inner_product(x, x, M))

def M_orthonormalize(A: hippylib.MultiVector, M: dolfin.Matrix) -> hippylib.MultiVector:
    num_vectors = A.nvec()
    Q = hippylib.MultiVector(A[0], num_vectors)
    R = numpy.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        R[i,i] = M_norm(A[i], M)
        Q[i][:] = A[i].get_local()/R[i,i]
        for j in range(i+1, num_vectors):
            R[i,j] = M_inner_product(Q[i], A[j], M)
            A[j][:] = A[j].get_local() - R[i,j]*Q[i].get_local()
    return Q

def function_inner_product(f: dolfin.Function, g: dolfin.Function) -> float:

    return dolfin.assemble(f*g*dolfin.dx)

def vector_weighted_inner_product(x: numpy.ndarray, y: numpy.ndarray, weight_matrix: numpy.ndarray) -> numpy.ndarray:

    return x.T @ (weight_matrix @ y)

def function_l2_norm(f: dolfin.Function) -> float:
    return dolfin.assemble((f**2)*dolfin.dx)**(1/2)

def function_l2_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    error = function_l2_norm(diff)
    return error
    
def function_relative_l2_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    rel_error = function_l2_norm(diff) / function_l2_norm(u)
    return  rel_error

def function_h10_norm(f: dolfin.Function) -> float:
    h10_norm = function_l2_norm(dolfin.grad(f))
    return h10_norm

def function_h10_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    error = function_h10_norm(diff)
    return error

def function_relative_h10_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    rel_error = function_h10_norm(diff) / function_h10_norm(u)
    return  rel_error

def function_h1_norm(f: dolfin.Function) -> float:
    l2_norm = function_l2_norm(f)
    grad_norm = function_l2_norm(dolfin.grad(f))
    h1_norm = numpy.sqrt(l2_norm ** 2 + grad_norm ** 2)
    return h1_norm

def function_h1_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    error = function_h1_norm(diff)
    return error

def function_relative_h1_error(u_hat: dolfin.Function, u: dolfin.Function, Vh: dolfin.FunctionSpace) -> float:
    diff = dolfin.project(u_hat - u, Vh)
    rel_error = function_h1_norm(diff) / function_h1_norm(u)
    return  rel_error
