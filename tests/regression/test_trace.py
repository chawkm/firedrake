from firedrake import *
import numpy as np

def projection(mesh):
    # Define the function spaces for testing (Pk CG, and Pk Trace)
    k = 2
    CG = FunctionSpace(mesh, "CG", k)
    TraceSpace = FunctionSpace(mesh, "HDiv Trace", k)

    HCG = FunctionSpace(mesh, "CG", k+3)

    W = CG*TraceSpace

    # Create mesh normal for trace integrals
    n = FacetNormal(mesh)

    # Define trial and test functions
    u, lambdar = TrialFunctions(W)
    v, gammar = TestFunctions(W)

    # Interpolate smooth function into the CG space
    f = Function(HCG)
    x = SpatialCoordinate(mesh)
    #f.interpolate(sin(x[0]*pi*2)*sin(x[1]*pi*2))
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))

    rstr = '+'

    # Construct the bilinear form
    a_dx = u*v*dx
    a_dS = lambdar*gammar*ds + avg(lambdar)*avg(gammar)*dS
    a = a_dx + a_dS

    # Construct the linear form
    L_dx = f*v*dx
    L_dS = f*gammar*ds + avg(f)*avg(gammar)*dS
    L = L_dx + L_dS

    # Solution
    w = Function(W)
    solve(a == L, w, solver_parameters={'ksp_rtol': 1e-14,
                                    'ksp_max_it': 10000})
    u_h, tr_h = w.split()

    #uherr = sqrt(assemble((u_h-f)*(u_h-f)*dx))
    uherr = sqrt(assemble((u_h - f)*(u_h - f)*ds + (avg(u_h) - avg(f))*(avg(u_h) - avg(f))*dS))
    trerr = sqrt(assemble((tr_h - f)*(tr_h - f)*ds + (avg(tr_h) - avg(f))*(avg(tr_h) - avg(f))*dS))

    return uherr, trerr

uherr = []
trerr = []
# Create a mesh
for r in range(8):
    res = 2**r
    mesh = UnitSquareMesh(res, res)

    e = projection(mesh)
    uherr.append(e[0])
    trerr.append(e[1])

uherr = np.array(uherr)
trerr = np.array(trerr)

print np.log(uherr[1:]/uherr[:-1])/np.log(0.5)
print np.log(trerr[1:]/trerr[:-1])/np.log(0.5)
print uherr
print trerr
