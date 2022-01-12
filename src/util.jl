"""
`block_split(n, max_blk_size)` returns `block_sizes` which divides `n` coordinates into block sizes. 

An alternative is to call `block_split` passing 
an `n×n` square matrix `Σ` or an `n` vector in place of `n`.
"""
function block_split end 

function block_split(n, max_blk_size)
    block_sizes = fill(max_blk_size, n÷max_blk_size)
    m = sum(block_sizes)
    if m < n
        append!(block_sizes, n - m)
    end
    block_sizes
end

function block_split(Σ::AbstractMatrix, max_blk_size) 
    n = LinearAlgebra.checksquare(Σ)
    block_split(n, max_blk_size)
end

function block_split(v::AbstractVector, max_blk_size) 
    n = length(v)
    block_split(n, max_blk_size)
end
