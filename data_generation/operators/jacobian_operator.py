import dolfin 

class JacobianOperator:
    def __init__(self, compile_form: callable, bcs0: list[dolfin.DirichletBC]):
        self.compile_form = compile_form
        self.bcs0 = bcs0

    def jacobian_action(self, m: dolfin.Function, u: dolfin.Function, input_functions: list[dolfin.Function]) -> list[dolfin.Function]:
        p = dolfin.TrialFunction(u.function_space())
        v = dolfin.TestFunction(u.function_space())
        R = self.compile_form(m,u,v)
        A = dolfin.assemble(dolfin.derivative(R,u,p))
        for bc in self.bcs0:
            bc.apply(A)
        solver = dolfin.LUSolver(A)
        output_functions = []
        for _,input_function in enumerate(input_functions):
            b = dolfin.assemble(-dolfin.derivative(R,m,input_function))
            for bc in self.bcs0:
                bc.apply(b)
            p = dolfin.Function(u.function_space())
            solver.solve(p.vector(), b)
            output_functions.append(p)
        return output_functions

    
    def adjoint_jacobian_action(self, m: dolfin.Function, u: dolfin.Function, input_functions: list[dolfin.Function]) -> list[dolfin.Vector]:
        q = dolfin.TrialFunction(u.function_space())
        v = dolfin.TestFunction(u.function_space())
        R = self.compile_form(m,u,q)
        A = dolfin.assemble(dolfin.derivative(R,u,v))
        for bc in self.bcs0:
            bc.apply(A)
        solver = dolfin.LUSolver(A)
        output_functions_projection = []
        for _,input_function in enumerate(input_functions):
            b = dolfin.assemble(dolfin.inner(input_function, v)*dolfin.dx)
            for bc in self.bcs0:
                bc.apply(b)
            q = dolfin.Function(u.function_space())
            solver.solve(q.vector(),b)
            v2 = dolfin.TestFunction(m.function_space())
            R2 = self.compile_form(m,u,q)
            b2 = dolfin.assemble(-dolfin.derivative(R2,m,v2))
            output_functions_projection.append(b2)
        return output_functions_projection