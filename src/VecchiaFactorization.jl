module VecchiaFactorization

import Base: size, getindex, permute!, invpermute!, show, rand, randn

using LinearAlgebra # BLAS.set_num_threads(1)
import LinearAlgebra: mul!, lmul!, ldiv!, \, /, *, inv, pinv, 
adjoint, transpose, Matrix, sqrt, Hermitian, Symmetric, cholesky

using SparseArrays: spdiagm
import SparseArrays: sparse

using ArrayLayouts # supposed to speed up mul! for Symmetric, etc...
				   # but for small matrices this doesn't appear to help

using BlockArrays: PseudoBlockArray,  Block, BlockArray, undef_blocks,
blocks, blocksizes, blockedrange, findblockindex, blockindex, mortar

export Ridiagonal, Midiagonal, Inv, Piv, sparse

# VecchiaFactor{T}
# ===========================================

## abstract type VecchiaFactor{T} <: AbstractMatrix{T} end
## abstract type VecchiaFactor{T} <: Factorization{T} end
abstract type VecchiaFactor{T} end

# Inv and Adj
# ===========================================
include("adj_inv.jl")


## parent(A::InvOrAdj_VF)  = A.parent

size(A::InvOrAdj_VF)    = reverse(size(A.parent))

function show(io::IO, ::MIME"text/plain", A::InvOrAdj_VF)
    print(io, typeof(A), "\n", A.parent)
end

pinv(A::InvOrAdjOrVecc_VF)       = inv(A)


# for operating on vectors the base methods are these (define them for each new type)
# mul!(w, ::VecchiaFactor, v)
# mul!(w, ::Adj_VF, v)
# ldiv!(w, ::VecchiaFactor, v)
# ldiv!(w, ::Adj_VF, v)

# * calls out to mul! and ldiv!
*(VF::A,   w::AbstractVector) where {A<:VecchiaFactor} = mul!(copy(w), VF, w)
*(VFᴴ::A,  w::AbstractVector) where {A<:Adj_VF}        = mul!(copy(w), VFᴴ, w)
*(iVF::A,  w::AbstractVector) where {A<:Inv_VF}        = ldiv!(iVF.parent, copy(w))
*(iVFᴴ::A, w::AbstractVector) where {A<:Inv_Adj_VF}    = ldiv!(iVFᴴ.parent, copy(w))

# \ calls out to inv * ...
\(VF::A,   w::AbstractVector) where {A<:VecchiaFactor} = inv(VF) * w
\(VFᴴ::A,  w::AbstractVector) where {A<:Adj_VF}        = inv(VFᴴ) * w
\(iVF::A,  w::AbstractVector) where {A<:Inv_VF}        = inv(iVF) * w
\(iVFᴴ::A, w::AbstractVector) where {A<:Inv_Adj_VF}    = inv(iVFᴴ) * w

# Chain products by creating a tuple
# ===========================================
include("op_chain.jl")

# these are operations on vectors ... op_chain operates on other operators
\(A::Adj, v::AbstractVector) = inv(A) * v
\(A::Inv, v::AbstractVector) = A.parent * v

# two specific VecchiaFactors
# ===========================================
include("mi_ri.jl")

include("pivot_type.jl") 

# constructing sparse or matrix equivalents
# ===========================================
include("sparse_matrix_show.jl")

# constructor Vecchia factorization
# ===========================================
include("vecchia_approx.jl")

# instantiating/construting the tridiagonal inverse
# ===========================================
include("vecchia_inv_access.jl")

# ===========================================
include("util.jl")

end
