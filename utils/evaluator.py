import numpy
import dolfin

from .misc import function_relative_l2_error, function_relative_h1_error, function_relative_h10_error

class Evaluator:
      def __init__(self, mesh):
         self.mesh = mesh

         self.Vh = dolfin.FunctionSpace(self.mesh, 'Lagrange', 1)
         self.Vh_vec = dolfin.VectorFunctionSpace(self.mesh, 'Lagrange', 1)

         self.dof_to_vertex_map = dolfin.dof_to_vertex_map(self.Vh)

      def evaluations_to_functions(self, evaluations: numpy.ndarray) -> list[dolfin.Function]:
         if len(evaluations.shape) != 2:
            raise ValueError('The dimension of evaluations must be 2.')
         if evaluations.shape[1] != self.Vh.dim():
            raise ValueError('The number of columns of evaluations must be the same as the dimension of the output function space.')
         functions = []
         num_functions = len(evaluations)
         for i in range(num_functions):
            u = dolfin.Function(self.Vh)
            u.vector().set_local(evaluations[i,:][self.dof_to_vertex_map])
            functions.append(u)
         return functions

      def evaluations_to_vector_functions(self, evaluations: numpy.ndarray) -> list[dolfin.Function]:
         if len(evaluations.shape) != 3:
            raise ValueError('The dimension of evaluations must be 3.')
         if evaluations.shape[1] != self.Vh_vec.sub(0).dim():
            raise ValueError('The number of columns of evaluations must be the same as the dimension of each component of the output vector function space.')
         functions = []
         num_functions = len(evaluations)
         output_dim = evaluations.shape[2]
         for i in range(num_functions):
            u = dolfin.Function(self.Vh_vec)
            components = u.split(deepcopy=True)
            for j in range(output_dim):
               components[j].vector().set_local(evaluations[i,:,j][self.dof_to_vertex_map])
               dolfin.assign(u.sub(j), components[j])
            functions.append(u)
         return functions

      def compute_function_avg_rel_err(self, output_functions: list[dolfin.Function], label_functions: list[dolfin.Function], norm_type: str) -> numpy.ndarray:
         """
         norm_type: 'L2', 'H1', 'H10'
         """
         if norm_type not in ['L2', 'H1', 'H10']:
            raise ValueError('The norm type must be "L2", "H1" or "H10".')
         if len(output_functions) != len(label_functions):
            raise ValueError('The number of output functions and label functions must be the same.')
         error = 0.0
         num_functions = len(output_functions)
         if output_functions[0].function_space().num_sub_spaces() == self.Vh.num_sub_spaces():
            for i in range(num_functions):
               if norm_type == 'L2':
                  error += function_relative_l2_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh)
               elif norm_type == 'H1':
                  error += function_relative_h1_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh)
               elif norm_type == 'H10':
                  error += function_relative_h10_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh)
         elif output_functions[0].function_space().num_sub_spaces() == self.Vh_vec.num_sub_spaces():
            for i in range(num_functions):
               if norm_type == 'L2':
                  error += function_relative_l2_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh_vec)
               elif norm_type == 'H1':
                  error += function_relative_h1_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh_vec)
               elif norm_type == 'H10':
                  error += function_relative_h10_error(u_hat=output_functions[i], u=label_functions[i], Vh=self.Vh_vec)
         else:
            raise NotImplementedError
         error /= num_functions
         return error

      def compute_matrix_avg_rel_err(self, outputs: numpy.ndarray, labels: numpy.ndarray) -> numpy.ndarray:
         """
         norm_type: 'fro'
         """
         num_matrices = outputs.shape[0]
         error = 0.0
         for i in range(num_matrices):
            error += numpy.sqrt(numpy.sum((outputs[i] - labels[i])**2))/numpy.sqrt(numpy.sum((labels[i])**2))
         error /= num_matrices
         return error
     

      def compute_multiple_avg_rel_err(self, outputs, labels): 
         if outputs['eval'].shape[2] == 1:
            output_functions = self.evaluations_to_functions(outputs['eval'][:,:,0])
            label_functions = self.evaluations_to_functions(labels['eval'][:,:,0])
         elif outputs['eval'].shape[2] == 2:
            output_functions = self.evaluations_to_vector_functions(outputs['eval'])
            label_functions = self.evaluations_to_vector_functions(labels['eval'])
         else:
            raise NotImplementedError

         avg_rel_l2_err = self.compute_function_avg_rel_err(output_functions, label_functions, norm_type='L2')
         avg_rel_h1_err = self.compute_function_avg_rel_err(output_functions, label_functions, norm_type='H1')
         avg_rel_fro_err = {
            'dm': self.compute_matrix_avg_rel_err(outputs['dm'], labels['dm']) if outputs['dm'] is not None else None,
            'dx': self.compute_matrix_avg_rel_err(outputs['dx'], labels['dx']) if outputs['dx'] is not None else None
         }
         
         return avg_rel_l2_err, avg_rel_h1_err, avg_rel_fro_err
