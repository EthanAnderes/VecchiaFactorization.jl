module VecchiaFactorization

using LinearAlgebra # BLAS.set_num_threads(1)

using BlockArrays: PseudoBlockArray, AbstractBlockMatrix, Block, 
blocks, blocksizes, blockedrange, findblockindex, blockindex, mortar

include("mi_ri_qi_diag.jl")
include("vecchia.jl")
include("lazyinv.jl")

end
