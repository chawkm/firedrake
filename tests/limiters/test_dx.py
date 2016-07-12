from firedrake import *
import numpy as np
import argparse


def run_test(mesh, outfile):
#    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

    V = FunctionSpace(mesh, "DG", 0)
    M = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    u0 = Expression(('1','0'))
    u = Function(M).interpolate(u0)
    
    iterations = 100
    dt = 1. / iterations

    phi = TestFunction(V)
    D = TrialFunction(V)

    n = FacetNormal(mesh)

    un = 0.5 * (dot(u, n) + abs(dot(u, n))) #upwind value

    a_mass = phi*D*dx
    a_int = dot(grad(phi), -u*D)*dx
    a_flux = dot(jump(phi), un('+')*D('+') - un('-')*D('-'))*dS
    arhs = a_mass - dt * (a_int + a_flux)

    dD1 = Function(V)
    D1 = Function(V)

    #changed
    x = SpatialCoordinate(mesh)
    D0 = Expression(sin(2*pi*x[0]))
    D = Function(V).interpolate(D0)
    D_old = Function(D)

    t = 0.0
    T = iterations*dt

    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})

    L2_0 = norm(D)
    Dbar_0 = assemble(D*dx)
    while t < (T - dt/2):
        D1.assign(D)
        solver.solve()
        D1.assign(dD1)

        solver.solve()
        D1.assign(0.75*D + 0.25*dD1)
        solver.solve()
        D.assign((1.0/3.0)*D + (2.0/3.0)*dD1)

        t += dt
        outfile.write(D1, time=t)

    L2_T = norm(D)
    Dbar_T = assemble(D*dx)

    # L2 norm decreases
    #assert L2_T < L2_0

    # Mass conserved
    #assert np.allclose(Dbar_T, Dbar_0)


if __name__ == '__main__':
    print "**********Start**********"
    print "Mesh size>: ",
    mesh_size = int(raw_input()) 
    run_test(PeriodicUnitSquareMesh(mesh_size,mesh_size), File("Periodic.pvd"))
    print "**********Done**********"
