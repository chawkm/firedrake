import pytest
import numpy as np

from firedrake import *


def test_real_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    f = Function(fs)

    f.dat.data[0] = 2.

    assert assemble(f*dx) == 2.0


def test_real_one_form_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    v = TestFunction(fs)

    assert assemble(v*dx).dat.data[0] == 1.0


def test_real_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    assert assemble(2*u*v*dx).M.values == 2.0


def test_real_nonsquare_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(rfs)
    v = TestFunction(cgfs)

    base_case = assemble(2*v*dx)
    m1 = assemble(2*u*v*dx)

    u = TrialFunction(cgfs)
    v = TestFunction(rfs)
    m2 = assemble(2*u*v*dx)

    np.testing.assert_almost_equal(base_case.dat.data,
                                   m1.M.values[:, 0])
    np.testing.assert_almost_equal(base_case.dat.data,
                                   m2.M.values[0, :])


def test_real_mixed_one_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    v, q = TestFunctions(mfs)

    A = assemble(v*dx + q*dx)

    qq = TestFunction(rfs)

    AA = assemble(qq*dx)

    np.testing.assert_almost_equal(A.dat.data[1],
                                   AA.dat.data)


def test_real_mixed_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    u, p = TrialFunctions(mfs)
    v, q = TestFunctions(mfs)

    m = assemble(u*v*dx + p*q*dx + u*q*dx + p*v*dx)

    qq = TestFunction(rfs)
    vv = TestFunction(cgfs)
    uu = TrialFunction(cgfs)

    m00 = assemble(uu*vv*dx)
    np.testing.assert_almost_equal(m00.M.values,
                                   m.M.blocks[0][0].values)
    m01 = assemble(uu*qq*dx)
    np.testing.assert_almost_equal(m01.M.values.T,
                                   m.M.blocks[0][1].values)
    np.testing.assert_almost_equal(m01.M.values,
                                   m.M.blocks[1][0].values)
    np.testing.assert_almost_equal(np.array([[1.]]),
                                   m.M.blocks[1][1].values)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
