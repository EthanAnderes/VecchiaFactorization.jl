"""
`block_split(n, max_blk_size)` returns `block_sizes` which divides `n` coordinates into block sizes. 
"""
function block_split(n, max_blk_size)
    block_sizes = fill(max_blk_size, n÷max_blk_size)
    m = sum(block_sizes)
    if m < n
        append!(block_sizes, n - m)
    end
    block_sizes
end

