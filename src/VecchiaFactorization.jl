module VecchiaFactorization

using LinearAlgebra # BLAS.set_num_threads(1)
using BlockArrays: PseudoBlockArray, AbstractBlockMatrix, Block, 
blocks, blocksizes, blockedrange, findblockindex, blockindex, mortar

abstract type VecchiaFactor{T} <: AbstractMatrix{T} end
const VeccFactorOrAdjoint{T} = Union{VecchiaFactor{T}, Adjoint{T,<:VecchiaFactor{T}}}

include("mi_ri_qi_diag.jl")
include("lazy_inv.jl")
include("pivot_type.jl")
include("op_chain.jl")
include("vecchia.jl")

end
