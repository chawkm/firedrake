cimport petsc4py.PETSc as PETSc
cimport mpi4py.MPI as MPI


cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    int PetscMalloc1(PetscInt,void*)
    int PetscFree(void*)
    int PetscSortInt(PetscInt,PetscInt[])

    int VecGetArray(PETSc.PetscVec, PetscScalar**)
    int VecGetArrayRead(PETSc.PetscVec, const PetscScalar**)
    int VecRestoreArray(PETSc.PetscVec, PetscScalar**)
    int VecRestoreArrayRead(PETSc.PetscVec, const PetscScalar**)

cdef extern from "hash.h":
    ctypedef long khiter_t
    ctypedef long khint_t
    struct khash_32_t
    ctypedef khash_32_t* hash_t
    hash_t kh_init(int)
    void kh_destroy(int, hash_t)
    void kh_clear(int, hash_t)
    void kh_resize(int, hash_t, int)
    khiter_t kh_put(int, hash_t, int, khiter_t*)
    khiter_t kh_get(int, hash_t, int)
    khiter_t kh_del(int, hash_t, khint_t)
    int kh_exist(hash_t, khint_t)
    int kh_key(hash_t, khint_t)
    int kh_val(hash_t, khint_t)
    void kh_set_val(hash_t, khiter_t, int)
    khiter_t kh_begin(hash_t)
    khiter_t kh_end(hash_t)
    khint_t kh_size(hash_t)

cdef extern from "petscdmplex.h":
    int DMPlexGetHeightStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)
    int DMPlexGetDepthStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)

    int DMPlexGetCone(PETSc.PetscDM,PetscInt,PetscInt*[])
    int DMPlexGetConeSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetSupport(PETSc.PetscDM,PetscInt,PetscInt*[])
    int DMPlexGetSupportSize(PETSc.PetscDM,PetscInt,PetscInt*)

    int DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])

cdef extern from "petscdmlabel.h":
    struct _n_DMLabel
    ctypedef _n_DMLabel* DMLabel "DMLabel"
    int DMLabelCreateIndex(DMLabel, PetscInt, PetscInt)
    int DMLabelHasPoint(DMLabel, PetscInt, PetscBool*)

cdef extern from "petscdm.h":
    int DMGetLabel(PETSc.PetscDM,char[],DMLabel*)

cdef extern from "petscis.h":
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    int PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)

cdef extern from "petscsf.h":
    int PetscSFBcastBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,)
    int PetscSFBcastEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*)
    int PetscSFReduceBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)
    int PetscSFReduceEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)