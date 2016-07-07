"""This is a symbolic language for algebraic tensor expressions.
This work is based on a template written by Lawrence Mitchell.

Written by: Thomas Gibson
"""

from __future__ import absolute_import
import numpy as np
from singledispatch import singledispatch
import operator
import itertools
import functools
import firedrake
import ufl
from coffee import base as AST
from coffee.visitor import Visitor


class Tensor(ufl.form.Form):
    """An abstract class for Tensor SLATE nodes.
    This tensor class also inherits directly from
    ufl.form for composability purposes.
    """

    children = ()
    id_num = 0

    def __init__(self, arguments, coefficients):
        self.id = Tensor.id_num
        self._arguments = tuple(arguments)
        self._coefficients = tuple(coefficients)
        shape = []
        shapes = {}
        for i, arg in enumerate(self._arguments):
            V = arg.function_space()
            shapeList = []
            for funcspace in V:
                shapeList.append(funcspace.fiat_element.space_dimension() *
                                 funcspace.dim)
            shapes[i] = tuple(shapeList)
            shape.append(sum(shapeList))
        self.shapes = shapes
        self.shape = tuple(shape)

    def arguments(self):
        return self._arguments

    def coefficients(self):
        return self._coefficients

    def __add__(self, other):
        return TensorAdd(self, other)

    def __sub__(self, other):
        return TensorSub(self, other)

    def __mul__(self, other):
        return TensorMul(self, other)

    def __neg__(self):
        return Negative(self)

    def __pos__(self):
        return Positive(self)

    @property
    def inv(self):
        return Inverse(self)

    @property
    def T(self):
        return Transpose(self)

    @property
    def operands(self):
        """Returns the objects which this object
        operates on.
        """
        return ()


class Scalar(Tensor):
    """An abstract class for scalar-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 0
        self.rank = r
        self.form = form
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=(),
                        coefficients=form.coefficients())

    def __str__(self):
        return "S_%d" % self.id

    __repr__ = __str__


class Vector(Tensor):
    """An abstract class for vector-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 1
        self.rank = r
        self.form = form
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=form.arguments(),
                        coefficients=form.coefficients())

    def __str__(self):
        return "V_%d" % self.id

    __repr__ = __str__


class Matrix(Tensor):
    """An abstract class for matrix-valued SLATE nodes."""

    __slots__ = ('rank', 'form')
    __front__ = ('rank', 'form')

    def __init__(self, form):
        r = len(form.arguments())
        assert r == 2
        self.rank = r
        self.form = form
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=self.form.arguments(),
                        coefficients=self.form.coefficients())

    def __str__(self):
        return "M_%d" % self.id

    __repr__ = __str__


class Inverse(Tensor):
    """An abstract class for the tensor inverse SLATE node."""

    __slots__ = ('children', )

    def __init__(self, tensor):
        assert len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1], \
            "Taking inverses only makes sense for rank 2 square tensors."
        self.children = tensor
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=reversed(tensor.arguments()),
                        coefficients=tensor.coefficients())

    def __str__(self):
        return "%s.inverse()" % self.children

    def __repr__(self):
        return "Inverse(%s)" % self.children

    @property
    def operands(self):
        return (self.children, )


class Transpose(Tensor):
    """An abstract class for the tensor transpose SLATE node."""

    __slots__ = ('children', )

    def __init__(self, tensor):
        self.children = tensor
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=reversed(tensor.arguments()),
                        coefficients=tensor.coefficients())

    def __str__(self):
        return "%s.transpose()" % self.children

    def __repr__(self):
        return "Transpose(%s)" % self.children

    @property
    def operands(self):
        return (self.children, )


class UnaryOp(Tensor):
    """An abstract SLATE class for unary operations on tensors.
    Such operations take only one operand, ie a single input.
    An example is the negation operator: ('Negative(A)' = -A).
    """

    __slots__ = ('children', )

    def __init__(self, tensor):
        self.children = tensor
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=tensor.arguments(),
                        coefficients=tensor.coefficients())

    def __str__(self, order_of_operation=None):
        ops = {operator.neg: '-',
               operator.pos: '+'}
        if (order_of_operation is None) or (self.order_of_operation >= order_of_operation):
            pars = lambda X: X
        else:
            pars = lambda X: "(%s)" % X

        return pars("%s%s" % (ops[self.operation], self.children__str__()))

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.children)

    @property
    def operands(self):
        return (self.children, )


class Negative(UnaryOp):
    """Class for the negation of a tensor object."""

    # Class variables for the negation operator
    order_of_operation = 1
    operation = operator.neg


class Positive(UnaryOp):
    """Class for the positive operation on a tensor."""

    # Class variables
    order_of_operation = 1
    operation = operator.pos


class BinaryOp(Tensor):
    """An abstract SLATE class for binary operations on tensors.
    Such operations take two operands, and returns a tensor-valued
    expression.
    """

    __slots__ = ('children', )

    def __init__(self, A, B):
        args = self.get_arguments(A, B)
        coeffs = self.get_coefficients(A, B)
        self.children = A, B
        Tensor.id_num += 1
        Tensor.__init__(self, arguments=args,
                        coefficients=coeffs)

    @classmethod
    def get_arguments(cls, A, B):
        pass

    @classmethod
    def get_coefficients(cls, A, B):
        # Remove duplicate coefficients in forms
        coeffs = []
        # 'set()' creates an iterable list of unique elements
        A_UniqueCoeffs = set(A.coefficients())
        for c in B.coefficients():
            if c not in A_UniqueCoeffs:
                coeffs.append(c)
        return tuple(list(A.coefficients()) + coeffs)

    def __str__(self, order_of_operation=None):
        ops = {operator.add: '+',
               operator.sub: '-',
               operator.mul: '*'}
        if (order_of_operation is None) or (self.order_of_operation >= order_of_operation):
            pars = lambda X: X
        else:
            pars = lambda X: "(%s)" % X
        operand1 = self.children[0].__str__()
        operand2 = self.children[1].__str__()
        result = "%s %s %s" % (operand1, ops[self.operation],
                               operand2)
        return pars(result)

    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__,
                               self.children[0],
                               self.children[1])


class TensorAdd(BinaryOp):
    """This class represents the binary operation of
    addition on tensor objects.
    """

    # Class variables for tensor addition
    order_of_operation = 1
    operation = operator.add

    @classmethod
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        assert A.shape == B.shape
        return A.arguments()


class TensorSub(BinaryOp):
    """This class represents the binary operation of
    subtraction on tensor objects.
    """

    # Class variables for tensor subtraction
    order_of_operation = 1
    operation = operator.sub

    @classmethod
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        assert A.shape == B.shape
        return A.arguments()


class TensorMul(BinaryOp):
    """This class represents the binary operation of
    multiplication on tensor objects.
    """

    # Class variables for tensor product
    order_of_operation = 2
    operation = operator.mul

    @classmethod
    def get_arguments(cls, A, B):
        # Scalars distribute over sums
        if isinstance(A, Scalar):
            return B.arguments()
        elif isinstance(B, Scalar):
            return A.arguments()
        # Check for function space type to perform contraction
        # over middle indices
        assert (A.arguments()[-1].function_space() ==
                B.arguments()[0].function_space())
        return A.arguments()[:-1] + B.arguments()[1:]
