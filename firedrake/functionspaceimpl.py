"""
This module provides the implementations of :class:`~.FunctionSpace`
and :class:`~.MixedFunctionSpace` objects, along with some utility
classes for attaching extra information to instances of these.
"""

from __future__ import absolute_import

import numpy
import weakref

import ufl

from pyop2 import op2
from tsfc.fiatinterface import create_element

from firedrake.functionspacedata import get_shared_data
from firedrake.petsc import PETSc
from firedrake import utils


class WithGeometry(ufl.FunctionSpace):
    """Attach geometric information to a :class:`~.FunctionSpace`.

    Function spaces on meshes with different geometry but the same
    topology can share data, except for their UFL cell.  This class
    facilitates that.

    Users should not instantiate a :class:`WithGeometry` object
    explicitly except in a small number of cases.

    :arg function_space: The topological function space to attach
        geometry to.
    :arg mesh: The mesh with geometric information to use.
    """
    def __init__(self, function_space, mesh):
        from firedrake.ufl_expr import reconstruct_element
        function_space = function_space.topological
        assert mesh.topology is function_space.mesh()
        assert mesh.topology is not mesh

        element = reconstruct_element(function_space.ufl_element(),
                                      cell=mesh.ufl_cell())
        super(WithGeometry, self).__init__(mesh, element)
        self.topological = function_space

        if function_space.parent is not None:
            self.parent = WithGeometry(function_space.parent, mesh)
        else:
            self.parent = None

    mesh = ufl.FunctionSpace.ufl_domain

    def ufl_function_space(self):
        """The :class:`ufl.FunctionSpace` this object represents."""
        return self

    def ufl_cell(self):
        """The :class:`ufl.Cell` this FunctionSpace is defined on."""
        return self.ufl_domain().ufl_cell()

    def split(self):
        """Split into a tuple of constituent spaces."""
        return tuple(WithGeometry(subspace, self.mesh())
                     for subspace in self.topological.split())

    def sub(self, i):
        return WithGeometry(self.topological.sub(i), self.mesh())

    def __eq__(self, other):
        try:
            return self.topological == other.topological and \
                self.mesh() is other.mesh()
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.topological)

    def __repr__(self):
        return "WithGeometry(%r, %r)" % (self.topological, self.mesh())

    def __str__(self):
        return "WithGeometry(%s, %s)" % (self.topological, self.mesh())

    def __iter__(self):
        for subspace in self.topological:
            yield WithGeometry(subspace, self.mesh())

    def __getitem__(self, i):
        return WithGeometry(self.topological[i], self.mesh())

    def __mul__(self, other):
        """Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        from firedrake.functionspace import MixedFunctionSpace
        return MixedFunctionSpace((self, other))

    def __getattr__(self, name):
        return getattr(self.topological, name)


class FunctionSpace(object):
    """A representation of a function space.

    A :class:`FunctionSpace` associates degrees of freedom with
    topological mesh entities.  The degree of freedom mapping is
    determined from the provided element.

    :arg mesh: The :func:`~.Mesh` to build the function space on.
    :arg element: The :class:`ufl.FiniteElementBase` describing the
        degrees of freedom.
    :kwarg name: An optional name for this :class:`FunctionSpace`,
        useful for later identification.

    The element can be a essentially any ufl FiniteElementBase, except
    for a :class:`ufl.MixedElement`, for which one should use the
    :class:`MixedFunctionSpace` constructor.

    To determine whether the space is scalar-, vector- or
    tensor-valued, one should inspect the :attr:`rank` of the
    resulting object.  Note that function spaces created on
    *intrinsically* vector-valued finite elements (such as the
    Raviart-Thomas space) have ``rank`` 0.

    .. warning::

       Users should build a :class:`FunctionSpace` directly, instead
       they should use the utility :func:`~.FunctionSpace` function,
       which provides extra error checking and argument sanitising.
    """
    def __init__(self, mesh, element, name=None):
        super(FunctionSpace, self).__init__()
        if type(element) is ufl.MixedElement:
            raise ValueError("Can't create FunctionSpace for MixedElement")
        fiat_element = create_element(element, vector_is_mixed=False)
        sdata = get_shared_data(mesh, fiat_element)
        # The function space shape is the number of dofs per node,
        # hence it is not always the value_shape.  Vector and Tensor
        # element modifiers *must* live on the outside!
        if type(element) in (ufl.VectorElement, ufl.TensorElement):
            self.shape = element.value_shape()
        else:
            self.shape = ()
        self._ufl_element = element
        self._shared_data = sdata
        self._mesh = mesh

        self.rank = len(self.shape)
        self.dim = numpy.prod(self.shape, dtype=int)
        self.name = name
        self.node_set = sdata.node_set
        self.dof_dset = op2.DataSet(self.node_set, self.shape or 1,
                                    name="%s_nodes_dset" % self.name)
        self.fiat_element = fiat_element
        self.extruded = sdata.extruded
        self.offset = sdata.offset
        self.bt_masks = sdata.bt_masks

    # These properties are overridden in ProxyFunctionSpaces, but are
    # provided by FunctionSpace so that we don't have to special case.
    index = None
    """The position of this space in its parent
    :class:`MixedFunctionSpace`, or ``None``."""

    parent = None
    """The parent space if this space was extracted from one, or ``None``."""

    component = None
    """The component of this space in its parent VectorElement space, or
    ``None``."""

    def __eq__(self, other):
        try:
            # FIXME: Think harder about equality
            return self.mesh() is other.mesh() and \
                self.dof_dset is other.dof_dset and \
                self.ufl_element() == other.ufl_element()
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @utils.cached_property
    def _dm(self):
        """A PETSc DM describing the data layout for this FunctionSpace."""
        dm = PETSc.DMShell().create()
        dm.setAttr('__fs__', weakref.ref(self))
        dm.setPointSF(self.mesh()._plex.getPointSF())
        dm.setDefaultSection(self._shared_data.global_numbering)
        # FIXME: The Vec could be shared between functionspaces with
        # the same dof_dset.
        with self.make_dat().vec_ro as v:
            dm.setGlobalVector(v.duplicate())
        return dm

    @utils.cached_property
    def _ises(self):
        return self.dof_dset.field_ises

    @utils.cached_property
    def cell_node_list(self):
        """A numpy array mapping mesh cells to function space nodes."""
        return self._shared_data.entity_node_lists[self.mesh().cell_set]

    @property
    def topological(self):
        """Function space on a mesh topology."""
        return self

    def mesh(self):
        return self._mesh

    def ufl_element(self):
        return self._ufl_element

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __repr__(self):
        return "FunctionSpace(%r, %r, name=%r)" % (self.mesh(),
                                                   self.ufl_element(),
                                                   self.name)

    def __str__(self):
        return "FunctionSpace(%s, %s, name=%s)" % (self.mesh(),
                                                   self.ufl_element(),
                                                   self.name)

    def split(self):
        """Split into a tuple of constituent spaces."""
        return (self, )

    def __getitem__(self, i):
        """Return the ith subspace."""
        if i != 0:
            raise IndexError("Only index 0 supported on a FunctionSpace")
        return self

    def sub(self, i):
        """Return a view into the ith component."""
        if self.rank != 1:
            raise ValueError("Can only take sub of FS with VectorElement")
        return ComponentFunctionSpace(self, i)

    def __mul__(self, other):
        """Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        from firedrake.functionspace import MixedFunctionSpace
        return MixedFunctionSpace((self, other))

    @property
    def node_count(self):
        """The number of global nodes in the function space.  If the
        :class:`FunctionSpace` has :attr:`rank` 0, this is equal to
        the :attr:`dof_count`, otherwise the :attr:`dof_count` is
        :attr:`dim` times the :attr:`node_count`."""
        return self.node_set.total_size

    @property
    def dof_count(self):
        """The number of global degrees of freedom in the function
        space. Cf. :attr:`node_count`."""
        return self.node_count*self.dim

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof_dset` of this :class:`.Function`."""
        return op2.Dat(self.dof_dset, val, valuetype, name, uid=uid)

    def cell_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.cell_node_map()
        else:
            parent = None

        sdata = self._shared_data
        return sdata.get_map(self,
                             self.mesh().cell_set,
                             self.fiat_element.space_dimension(),
                             bcs,
                             "cell_node",
                             self.offset,
                             parent)

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.interior_facet_node_map()
        else:
            parent = None

        sdata = self._shared_data
        offset = self.cell_node_map().offset
        map = sdata.get_map(self,
                            self.mesh().interior_facets.set,
                            2*self.fiat_element.space_dimension(),
                            bcs,
                            "interior_facet_node",
                            numpy.append(offset, offset),
                            parent)
        map.factors = (self.mesh().interior_facets.facet_cell_map,
                       self.cell_node_map())
        return map

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.exterior_facet_node_map()
        else:
            parent = None

        sdata = self._shared_data
        return sdata.get_map(self,
                             self.mesh().exterior_facets.set,
                             self.fiat_element.space_dimension(),
                             bcs,
                             "exterior_facet_node",
                             self.offset,
                             parent)

    def exterior_facet_boundary_node_map(self, method):
        """The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.

        :arg method: The method for determining boundary nodes. See
            :class:`~.bcs.DirichletBC`.
        """
        return self._shared_data.exterior_facet_boundary_node_map(self, method)

    def bottom_nodes(self, method='topological'):
        """Return a list of the bottom boundary nodes of the extruded mesh.
        The bottom mask is applied to every bottom layer cell to get the
        dof ids."""
        if self.bt_masks is None:
            raise ValueError("Doesn't make sense on non extruded space.")
        try:
            mask = self.bt_masks[method][0]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        return numpy.unique(self.cell_node_list[:, mask])

    def top_nodes(self, method='topological'):
        """Return a list of the top boundary nodes of the extruded mesh.
        The top mask is applied to every top layer cell to get the dof ids."""
        if self.bt_masks is None:
            raise ValueError("Doesn't make sense on non extruded space.")
        try:
            mask = self.bt_masks[method][1]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        voffs = self.offset.take(mask)*(self.mesh().layers-2)
        return numpy.unique(self.cell_node_list[:, mask] + voffs)


class MixedFunctionSpace(object):
    """A function space on a mixed finite element.

    This is essentially just a bag of individual
    :class:`FunctionSpace` objects.

    :arg spaces: The constituent spaces.
    :kwarg name: An optional name for the mixed space.

    .. warning::

       Users should not build a :class:`MixedFunctionSpace` directly,
       but should instead use the functional interface provided by
       :func:`MixedFunctionSpace`.
    """
    def __init__(self, spaces, name=None):
        super(MixedFunctionSpace, self).__init__()
        self._spaces = tuple(IndexedFunctionSpace(i, s, self)
                             for i, s in enumerate(spaces))
        self._ufl_element = ufl.MixedElement(*[s.ufl_element() for s
                                               in spaces])
        self.name = name or "_".join(str(s.name) for s in spaces)
        self._subspaces = []
        self._mesh = spaces[0].mesh()

    # These properties are so a mixed space can behave like a normal FunctionSpace.
    index = None
    component = None
    parent = None
    rank = 1

    def mesh(self):
        return self._mesh

    @property
    def topological(self):
        """Function space on a mesh topology."""
        return self

    def ufl_element(self):
        """The :class:`ufl.Mixedelement` this space represents."""
        return self._ufl_element

    def __eq__(self, other):
        if type(other) is not MixedFunctionSpace:
            return False
        return all(s == o for s, o in zip(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def split(self):
        """The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    def sub(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self[i]

    def num_sub_spaces(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self)

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self._spaces)

    def __getitem__(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def __iter__(self):
        for s in self._spaces:
            yield s

    def __repr__(self):
        return "MixedFunctionSpace(%s, name=%r)" % \
            (", ".join(repr(s) for s in self), self.name)

    def __str__(self):
        return "MixedFunctionSpace(%s)" % ", ".join(str(s) for s in self)

    @property
    def dim(self):
        """Return the sum of the :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return sum(fs.dim for fs in self._spaces)

    @property
    def node_count(self):
        """Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @property
    def dof_count(self):
        """Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of one or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.MixedDataSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self, bcs=None):
        """A :class:`pyop2.MixedMap` from the :attr:`Mesh.cell_set` of the
        underlying mesh to the :attr:`node_set` of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.cell_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.MixedMap` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.interior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.exterior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.MixedMap` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return op2.MixedMap(s.exterior_facet_boundary_node_map for s in self._spaces)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, "%s[cmpt-%d]" % (name, i), utils._new_uid())
                            for i, (s, v) in enumerate(zip(self._spaces, val)))

    @utils.cached_property
    def _dm(self):
        """A PETSc DM describing the data layout for fieldsplit solvers."""
        dm = PETSc.DMShell().create()
        dm.setAttr('__fs__', weakref.ref(self))
        dm.setCreateFieldDecomposition(self.create_field_decomp)
        dm.setCreateSubDM(self.create_subdm)
        # FIXME: The Vec could be shared between functionspaces with
        # the same dof_dset.
        with self.make_dat().vec_ro as v:
            dm.setGlobalVector(v.duplicate())
        return dm

    @utils.cached_property
    def _ises(self):
        return self.dof_dset.field_ises

    @classmethod
    def create_subdm(cls, dm, fields, *args, **kwargs):
        W = dm.getAttr('__fs__')()
        if len(fields) == 1:
            # Subspace is just a single FunctionSpace.
            subspace = W[fields[0]]
        else:
            # Need to build an MFS for the subspace
            subspace = MixedFunctionSpace([W[f] for f in fields])
        # Sub-DM is just the DM belonging to the subspace.
        subdm = subspace._dm
        # Keep hold of strong reference to created subspace (given we
        # only hold a weakref in the shell DM)
        W._subspaces.append(subspace)
        # Index set mapping from W into subspace.
        iset = PETSc.IS().createGeneral(numpy.concatenate([W._ises[f].indices
                                                           for f in fields]))
        return iset, subdm

    @classmethod
    def create_field_decomp(cls, dm, *args, **kwargs):
        W = dm.getAttr('__fs__')()
        # Don't pass split number if name is None (this way the
        # recursively created splits have the names you want)
        names = [s.name for s in W]
        dms = [V._dm for V in W]
        return names, W._ises, dms


class ProxyFunctionSpace(FunctionSpace):
    """A :class:`FunctionSpace` that one can attach extra properties to.

    :arg mesh: The mesh to use.
    :arg element: The UFL element.
    :arg name: The name of the function space.
    """
    def __new__(cls, mesh, element, name=None):
        topology = mesh.topology
        self = super(ProxyFunctionSpace, cls).__new__(cls, topology, element, name=name)
        if mesh is not topology:
            return WithGeometry(self, mesh)
        else:
            return self

    def __repr__(self):
        return "%sProxyFunctionSpace(%r, %r, name=%r, index=%r, component=%r)" % \
            (str(self.identifier).capitalize(),
             self.mesh(),
             self.ufl_element(),
             self.name,
             self.index,
             self.component)

    def __str__(self):
        return "%sProxyFunctionSpace(%s, %s, name=%s, index=%s, component=%s)" % \
            (str(self.identifier).capitalize(),
             self.mesh(),
             self.ufl_element(),
             self.name,
             self.index,
             self.component)

    identifier = None
    """An optional identifier, for debugging purposes."""

    no_dats = False
    """Can this proxy make :class:`pyop2.Dat` objects"""

    def make_dat(self, *args, **kwargs):
        """Create a :class:`pyop2.Dat`.

        :raises ValueError: if :attr:`no_dats` is ``True``.
        """
        if self.no_dats:
            raise ValueError("Can't build Function on %s function space" % self.typ)
        return super(ProxyFunctionSpace, self).make_dat(*args, **kwargs)


def IndexedFunctionSpace(index, space, parent):
    """Build a new FunctionSpace that remembers it is a particular
    subspace of a :class:`MixedFunctionSpace`.

    :arg index: The index into the parent space.
    :arg space: The subspace to represent
    :arg parent: The parent mixed space.
    :returns: A new :class:`ProxyFunctionSpaceBase` with index and parent
        set.
    """
    new = ProxyFunctionSpace(space.mesh(), space.ufl_element(),
                             name=space.name)
    new.index = index
    new.parent = parent
    new.identifier = "indexed"
    return new


def ComponentFunctionSpace(parent, component):
    """Build a new FunctionSpace that remembers it represents a
    particular component.  Used for applying boundary conditions to
    components of a :func:`VectorFunctionSpace`.

    :arg parent: The parent space (a FunctionSpace with a
        VectorElement).
    :arg component: The component to represent.
    :returns: A new :class:`ProxyFunctionSpaceBase` with the component set.
    """
    element = parent.ufl_element()
    assert type(element) is ufl.VectorElement
    if not (0 <= component < parent.dim):
        raise IndexError("Invalid component %d. not in [0, %d)" %
                         (component, parent.dim))
    if component > 2:
        raise NotImplementedError("Indexing component > 2 not implemented")
    new = ProxyFunctionSpace(parent.mesh(), element.sub_elements()[0],
                             name=parent.name)
    new.identifier = "component"
    new.component = component
    new.parent = parent
    new.no_dats = True
    return new
