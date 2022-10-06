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



"""
`_posdef(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}`

Clamp the eigenvalues less than ϵ
"""
function _posdef(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
    Tr  = real(T)
    F   = eigen(Sym_or_Hrm(M))
    λs  = F.values
    λs′ = clamp.(real.(F.values), ϵ, Tr(Inf))
    M′  = F.vectors * Diagonal(λs′) * F.vectors'
    # M′  = F.vectors * Diagonal(λs′) / F.vectors
    return M′
end
# _posdef(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
#     Tr  = real(T)
#     F   = eigen(Sym_or_Hrm(M), ϵ, Tr(Inf))
#     M′  = F.vectors * Diagonal(F.values .- ϵ) / F.vectors
#     M′ += ϵ*I
#     return Matrix(Sym_or_Hrm(M′))
# end

"""
`_posdef_sqrt(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}`

Clamp the eigenvalues less than ϵ and take the square root
"""
function _posdef_sqrt(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
    Tr  = real(T)
    F   = eigen(Sym_or_Hrm(M))
    λs  = F.values
    λs′ = clamp.(real.(F.values), ϵ, Tr(Inf))
    M′  = F.vectors * Diagonal(sqrt.(λs′)) * F.vectors'
    # M′  = F.vectors * Diagonal(sqrt.(λs′)) / F.vectors
    return M′
end



"""
`force_posdef(M::Symmetric{T}, ϵ=10eps(T))`

like `_posdef` but preserves the symmetric or hermitian types
"""
function force_posdef(M::Symmetric{T}, ϵ=100eps(T)) where {T<:Real}
    uplo = LinearAlgebra.sym_uplo(M.uplo)
    return Symmetric(_posdef(M, ϵ), uplo)
end

function force_posdef(M::Hermitian{C}, ϵ=100eps(T)) where {T, C<:Complex{T}}
    uplo = LinearAlgebra.sym_uplo(M.uplo)
    return Hermitian(_posdef(M, ϵ), uplo)
end

function force_posdef(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
    return Matrix(Sym_or_Hrm(_posdef(M, ϵ)))
end



"""
`force_posdef_sqrt(M::Symmetric{T}, ϵ=10eps(T))`

like `_posdef_sqrt` but preserves the symmetric or hermitian types
"""
function force_posdef_sqrt(M::Symmetric{T}, ϵ=100eps(T)) where {T<:Real}
    uplo = LinearAlgebra.sym_uplo(M.uplo)
    return Symmetric(_posdef_sqrt(M, ϵ), uplo)
end

function force_posdef_sqrt(M::Hermitian{C}, ϵ=100eps(T)) where {T, C<:Complex{T}}
    uplo = LinearAlgebra.sym_uplo(M.uplo)
    return Hermitian(_posdef_sqrt(M, ϵ), uplo)
end

function force_posdef_sqrt(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
    return Matrix(Sym_or_Hrm(_posdef_sqrt(M, ϵ)))
end


"""
`force_chol(M::AbstractMatrix{T}, ϵ=10eps(real(T)))`

Clamp the eigenvalues less than ϵ
"""
function force_chol(M::AbstractMatrix{T}, ϵ=100eps(real(T))) where {T}
    M′ = Sym_or_Hrm(M)
    C  = cholesky(M′; check = false)
    if isposdef(C)
        return C 
    else
        return cholesky(force_posdef(M′, ϵ))
    end 
end 

# used in vecchia_inv_access
inv_chol_L(M)    = inv(force_chol(M).L)
inv_chol_U(M)    = inv(force_chol(M).U)
inv_with_chol(M) = inv(force_chol(M))

Sym_or_Hrm(A::AbstractMatrix{<:Real},    uplo=:L) = Symmetric(A, uplo)
Sym_or_Hrm(A::AbstractMatrix{<:Complex}, uplo=:L) = Hermitian(A, uplo)
Sym_or_Hrm(A::Symmetric{T}) where {T} = A
Sym_or_Hrm(A::Hermitian{T}) where {T} = A

Typ_Sym_or_Hrm(::Type{T}) where {T<:Real}     = Symmetric{T, Matrix{T}}
Typ_Sym_or_Hrm(::Type{T}) where {T<:Complex}  = Hermitian{T, Matrix{T}}



