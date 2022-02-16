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

function mortar_Bidiagonal_fill(c::T, blk_sizes) where T<:Number
    N = length(blk_sizes)
    M0 = Matrix{T}[fill(c, blk_sizes[ic], blk_sizes[ic]) for ic=1:N]
    M⁻1 = Matrix{T}[fill(c, blk_sizes[ic+1], blk_sizes[ic]) for ic=1:N-1]
    mortar(Bidiagonal(M0,M⁻1,:L))
end 

function mortar_Tridiagonal_fill(c::T, blk_sizes) where T<:Number
    N = length(blk_sizes)
    M0 = Matrix{T}[fill(c, blk_sizes[ic], blk_sizes[ic]) for ic=1:N]
    M⁻1 = Matrix{T}[fill(c, blk_sizes[ic+1], blk_sizes[ic]) for ic=1:N-1]
    M⁺1 = Matrix{T}[fill(c, blk_sizes[ic], blk_sizes[ic+1]) for ic=1:N-1]
    mortar(Tridiagonal(M⁻1, M0, M⁺1))
end 

function initalize_bidiag_lblks(::Type{T}, blk_sizes) where T
    N = length(blk_sizes)
    B = BlockArray{T}(undef_blocks, blk_sizes, blk_sizes)
    for ic=1:N
        B[Block(ic,ic)] = zeros(T, blk_sizes[ic], blk_sizes[ic])
        if ic < N 
            B[Block(ic+1,ic)] = zeros(T, blk_sizes[ic+1], blk_sizes[ic])
        end 
    end 
    B 
end 
