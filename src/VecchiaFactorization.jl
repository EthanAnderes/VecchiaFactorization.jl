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

# Interface: define each one of these for each new VecchiaFactorization type
# mul!(w, ::VecchiaFactor, v)
# mul!(w, ::Adj_VF, v)
# ldiv!(w, ::VecchiaFactor, v)
# ldiv!(w, ::Adj_VF, v)

# * calls out to mul! and ldiv!
*(VF::VecchiaFactor, w::AbstractVector) = mul!(copy(w), VF, w)
*(VFᴴ::Adj_VF,       w::AbstractVector) = mul!(copy(w), VFᴴ, w)
*(iVF::Inv_VF,       w::AbstractVector) = ldiv!(iVF.parent, copy(w))
*(iVFᴴ::Inv_Adj_VF,  w::AbstractVector) = ldiv!(iVFᴴ.parent, copy(w))

# \ calls out to inv * ...
\(VF::VecchiaFactor, w::AbstractVector) = inv(VF) * w
\(VFᴴ::Adj_VF,       w::AbstractVector) = inv(VFᴴ) * w
\(iVF::Inv_VF,       w::AbstractVector) = inv(iVF) * w
\(iVFᴴ::Inv_Adj_VF,  w::AbstractVector) = inv(iVFᴴ) * w

# Can we put this someplace else??
size(A::InvOrAdj_VF)       = reverse(size(A.parent))
pinv(A::InvOrAdjOrVecc_VF) = inv(A)

# Chain products by creating a tuple
# ===========================================
include("op_chain.jl")

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
