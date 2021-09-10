module VecchiaFactorization

using LinearAlgebra # BLAS.set_num_threads(1)
using BlockArrays: PseudoBlockArray, AbstractBlockMatrix, Block, 
blocks, blocksizes, blockedrange, findblockindex, blockindex, mortar

# slated for removal
using BlockBandedMatrices: BlockDiagonal, BlockBidiagonal
using FillArrays: Eye

import LinearAlgebra: mul!, lmul!, ldiv!, \, /, *, inv, pinv, 
adjoint, transpose, Matrix

import Base: size, getindex, permute!, invpermute!, parent, show, 
replace_in_print_matrix, rand, randn

export Inv, Ridiagonal, Qidiagonal, Midiagonal, Piv,
Vecchia, InvVecchia, VecchiaPivoted, InvVecchiaPivoted

# VecchiaFactor{T}
# ===========================================

abstract type VecchiaFactor{T} <: AbstractMatrix{T} end

# Inv which compliments Adjoint
# ===========================================
include("lazy_inv.jl")

const Adj_VecchiaFactor{T}     = Adjoint{<:Any,<:VecchiaFactor}
const Inv_VecchiaFactor{T}     = Inv{T,<:VecchiaFactor}
const Inv_Adj_VecchiaFactor{T} = Inv{T,<:Adj_VecchiaFactor}
const InvOrAdjOrVecc = Union{VecchiaFactor, Adj_VecchiaFactor, Inv_VecchiaFactor, Inv_Adj_VecchiaFactor}

# inv creats and Inv generically (you can bypass these)
inv(A::VecchiaFactor)         = Inv(A)    # -> Inv_VecchiaFactor
inv(A::Inv_VecchiaFactor)     = A.parent  # -> VecchiaFactor
inv(A::Adj_VecchiaFactor)     = Inv(A)    # -> Inv_Adj_VecchiaFactor
inv(A::Inv_Adj_VecchiaFactor) = A.parent  # -> Adj_VecchiaFactor
pinv(A::InvOrAdjOrVecc) = inv(A)

# adjoint automatically creats an Adjoint (you can bypass these)
# adjoint(A::VecchiaFactor)     # automatic -> Adj_VecchiaFactor
adjoint(A::Inv_VecchiaFactor)     = Inv(adjoint(A.parent))  # -> Inv_Adj_VecchiaFactor
adjoint(A::Adj_VecchiaFactor)     = A.parent               # -> VecchiaFactor
adjoint(A::Inv_Adj_VecchiaFactor) = Inv(adjoint(A.parent)) # -> Inv_VecchiaFactor


# for operating on vectors the base methods are these (define them for each new type)
# mul!(w, ::VecchiaFactor, v)
# mul!(w, ::Adj_VecchiaFactor, v)
# ldiv!(w, ::VecchiaFactor, v)
# ldiv!(w, ::Adj_VecchiaFactor, v)

# * calls out to mul! and ldiv!
*(VF::A,   w::AbstractVector) where {A<:VecchiaFactor}         = mul!(copy(w), VF, w)
*(VFᴴ::A,  w::AbstractVector) where {A<:Adj_VecchiaFactor}     = mul!(copy(w), VFᴴ, w)
*(iVF::A,  w::AbstractVector) where {A<:Inv_VecchiaFactor}     = ldiv!(iVF.parent, copy(w))
*(iVFᴴ::A, w::AbstractVector) where {A<:Inv_Adj_VecchiaFactor} = ldiv!(iVFᴴ.parent, copy(w))

# \ calls out to inv * ...
\(VF::A,   w::AbstractVector) where {A<:VecchiaFactor}         = inv(VF) * w
\(VFᴴ::A,  w::AbstractVector) where {A<:Adj_VecchiaFactor}     = inv(VFᴴ) * w
\(iVF::A,  w::AbstractVector) where {A<:Inv_VecchiaFactor}     = inv(iVF) * w
\(iVFᴴ::A, w::AbstractVector) where {A<:Inv_Adj_VecchiaFactor} = inv(iVFᴴ) * w

# Chain products by creating a tuple
# ===========================================
# the second argument type below is unionall 
# ... this allows you to bipass for a specific InvOrAdjOrVecc

function *(O1::A, O2::InvOrAdjOrVecc) where A<:InvOrAdjOrVecc 
    tuple(O1, O2)
end

function \(O1::A, O2::InvOrAdjOrVecc) where A<:InvOrAdjOrVecc 
    inv(O1) * O2
end

function /(O1::A, O2::InvOrAdjOrVecc) where A<:InvOrAdjOrVecc 
    O1 * inv(O2) 
end

# the remaining chain rules
include("op_chain.jl")

# two specific VecchiaFactors
# ===========================================
include("mi_ri_qi_diag.jl")
include("pivot_type.jl") # TODO: standardize this as a Vecchia factor

#-------------------------
include("vecchia.jl")

end
