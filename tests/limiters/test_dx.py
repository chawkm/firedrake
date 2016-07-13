from matplotlib import pyplot as plt
from itertools import izip
from firedrake import *
import numpy as np
import argparse
import math

def estimate_n(e1, e2, d1, d2):
    return math.log(e1 / e2) / math.log(d1 / d2)

def plot(args, title, xaxis, yaxis, filename, start=20, end=20):
    print "Generating image"

    error = True
    if filename != 'error.png':
        error = False

    xs, ys, err = [], [], []

    func = None
    if error: func = ys.append
    else:     func = err.append

    for i in xrange(end):
        xs.append(1./(start+i))
        func(run_test(PeriodicUnitSquareMesh(start + i, start + i),
                      File("Periodic.pvd"), args.iterations))
        if args.verbose:
            print i, "/", end

    if not error: #then estimate n at each point
        ys = [estimate_n(*args) for args in izip(err, err[1:], xs, xs[1:])]

    #define graph
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(title)
    if error:
        plt.plot(xs, ys)
    else:
        plt.plot(xs[1:],ys)
    print "Saving png to:", filename
    plt.savefig(filename)
    plt.show()


def run_test(mesh, outfile, iterations):
#    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

    V = FunctionSpace(mesh, "DG", 0)
    M = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    u0 = Expression(('1','0'))
    u = Function(M).interpolate(u0)

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

    #Return numerical errror from starting position
    diff = assemble((D - D_old)**2 * dx) ** 0.5
    return diff

    # L2 norm decreases
    #assert L2_T < L2_0

    # Mass conserved
    #assert np.allclose(Dbar_T, Dbar_0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Output numerical error \
                                                  in simple advection process")
    parser.add_argument("-verbose", action="store_true",
                        help="Increases output verbosity")
    parser.add_argument("-ploterror", action="store_true",
                        help="Plots and saves image of error against mesh size")
    parser.add_argument("-plotn", action="store_true",
                        help="Plots and saves image of estimation of n against dx")
    parser.add_argument("iterations", help="Set the number of iterations", type=int,
                        default=100, nargs='?')
    parser.add_argument("mesh_size", help="Set the mesh size to use", type=int,
                        default=30, nargs='?')
    args = parser.parse_args()

    if args.verbose:
        print "**********START**********"

    mesh_size = args.mesh_size

    if args.verbose and not (args.ploterror or args.plotn):
        print "iterations: ", args.iterations
        print "mesh_size:  ", mesh_size
        print "dt:         ", "1 /", args.iterations

#   Plots and saves error vs mesh_size
    if args.ploterror:
        plot(args, xaxis='dx', yaxis='Error',
             title='Error as mesh size decreases (dt=1/100)',
             filename='error.png')

#   Plots and saves n vs dx
    if args.plotn:
        plot(args, xaxis='dx', yaxis='n = log(e1/e2) / log(dx1/dx2)',
              title='Estimating n whilst changing dx (dt=1/100)',
              filename='estimate_n.png')

    if args.verbose:
        print "**********FIN**********"
