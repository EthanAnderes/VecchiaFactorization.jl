# Midiagonal, Ridiagonal matrices 
# ============================================================

"""
R <: Ridiagonal ≡ R₂ ⋯ Rₙ₋₁Rₙ is block bidiagonal on subdiagonal
M <: Midiagonal ≡ block_diag(M₁, ⋯ ,Mₙ) is block diagonal

with R.data == [R₂, …, Rₙ]
with M.data == [M₁, ⋯ ,Mₙ]
"""

struct Ridiagonal{T,M<:AbstractMatrix{T}} <: VecchiaFactor{T}
    data::Vector{M} # sub diagonal matrices stored along the last dimension
end

struct Midiagonal{T,M<:AbstractMatrix{T}} <: VecchiaFactor{T}
    data::Vector{M} # diagonal matrices stored along the last dimension
end

const MiRi{T,M}    = Union{Midiagonal{T,M}, Ridiagonal{T,M}}

# Interface for VecchiaFactors
# ===================================

# testing the removal of these methods ... 
# by doing so it seems to allow more specialization
# inv(M::Midiagonal)  = Midiagonal(map(x->inv(factorize(Sym_or_Hrm(x))), M.data)) # testing this new version
# pinv(M::Midiagonal)  = Midiagonal(map(x->pinv(Sym_or_Hrm(x)), M.data)) 

adjoint(M::Midiagonal)    = Midiagonal(map(adjoint, M.data)) 
transpose(M::Midiagonal)  = Midiagonal(map(transpose, M.data)) 

# slated for removal ... 
function posdef_inv(M::Midiagonal{T}) where {T} 
    M′ = map(M.data) do Md 
        _Md = Sym_or_Hrm(Matrix(Md))
        chol_Md = cholesky(_Md; check=false)
        if issuccess(chol_Md)
            return inv(chol_Md)
        else
            return zeros(T, size(_Md))
        end
    end
    Midiagonal(M′)
end

# merge products of Midiagonals
*(M1::Midiagonal, M2::Midiagonal) = Midiagonal(map(*, M1.data, M2.data))

# ldiv!
# ---------------------------------

# inv(M) * w
function ldiv!(M::Midiagonal{T}, w::AbstractVector{Q})  where {T,Q}
    rw      = similar(w,promote_type(T,Q))
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        copyto!(rwB[i], M.data[i] \ wB[i])
    end
    return rw
end

# inv(M') * w
function ldiv!(Mᴴ::Adj{<:Any,<:Midiagonal{T}}, w::AbstractVector{Q})  where {T,Q}
    M       = Mᴴ.parent
    rw      = similar(w,promote_type(T,Q))
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        copyto!(rwB[i], M.data[i]' \ wB[i])
    end
    return rw
end

# inv(R) * w ≡ (R₂ ⋯ RₙRₙ₋₁)⁻¹ * w
function ldiv!(R::Ridiagonal, w::AbstractVector{T}) where T
    wbB = block_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i+1], R.data[i], wbB[i], -1, true)      
    end
    return w
end

# inv(R)' * w ≡ (Rₙ'Rₙ₋₁' ⋯ R₂')⁻¹ * w
function ldiv!(Rᴴ::Adj{<:Any,<:Ridiagonal}, w::AbstractVector{T}) where T
    R   = Rᴴ.parent
    wbB = block_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i], R.data[i]', wbB[i+1], -1, true)     
    end
    return w
end


# lmul!
# ---------------------------------

# R * w ≡ (R₂ ⋯ RₙRₙ₋₁) * w
function lmul!(R::Ridiagonal, w::AbstractVector{T}) where T
    wbB = block_array(R, w)
    for i in length(wbB)-1:-1:1
        mul!(wbB[i+1], R.data[i], wbB[i], true, true)    
    end
    return w
end

# R' * w ≡ (Rₙ'Rₙ₋₁' ⋯ R₂')*w
function lmul!(Rᴴ::Adj{<:Any, <:Ridiagonal}, w::AbstractVector{T}) where T
    R   = Rᴴ.parent
    wbB = block_array(R, w)
    for i in 1:length(wbB)-1
        mul!(wbB[i], R.data[i]', wbB[i+1], true, true)       
    end
    return w
end


# 3 and 5 arg mul! 
# ---------------------------------

function mul!(rw::AbstractVector, R::A, w::AbstractVector) where {A<:Union{Ridiagonal, Adj{<:Any,<:Ridiagonal}}}
    copyto!(rw,w)
    lmul!(R, rw)
end

function mul!(rw::AbstractVector, M::A, w::AbstractVector) where {A<:Midiagonal}
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        mul!(rwB[i], M.data[i], wB[i])
    end
    return rw
end

function mul!(rw::AbstractVector, Mᴴ::A, w::AbstractVector) where {A<:Adj{<:Any,<:Midiagonal}}
    M = Mᴴ.parent
    rwB, wB = block_array(M, rw, w)
    for i = 1:length(wB)
        mul!(rwB[i], M.data[i]', wB[i])
    end
    return rw
end


# Non-interface methods
# ================================================

# TODO: where are these used ???
Ridiagonal(A::Ridiagonal) = A
Midiagonal(A::Midiagonal) = A
Ridiagonal{T}(A::Ridiagonal{T}) where {T} = A
Midiagonal{T}(A::Midiagonal{T}) where {T} = A
Ridiagonal{T}(A::Ridiagonal) where {T} = Ridiagonal{T}(A.data)
Midiagonal{T}(A::Midiagonal) where {T} = Midiagonal{T}(A.data)

nblocks(M::Midiagonal) = length(M.data)
nblocks(R::Ridiagonal) = length(R.data)+1

block_size(M::Midiagonal, d) = 0 < d::Integer <= 2 ? size.(M.data, 1) : 1
function block_size(R::Ridiagonal, d)
    if 0 < d::Integer <= 2
        bs = size.(R.data, 2)
        append!(bs, size(R.data[end],1))
    else 
        return 1 
    end
end 

# used to prep incoming arrays for lmul! or ldiv!
function block_array(MR::MiRi, w::AbstractVector)
    bs  = block_size(MR,2)
    blocks(PseudoBlockArray(w, bs))
end
function block_array(MR::MiRi, w::AbstractVector, v::AbstractVector)
    bs  = block_size(MR,2)
    blocks(PseudoBlockArray(w, bs)), blocks(PseudoBlockArray(v, bs))
end

# appears to be useful for generating random Midiagonals or Ridiagonals
sizes_from_blocksides(::Type{<:Midiagonal}, bs::Vector{Int}) = [(bs[i],bs[i]) for i=1:length(bs)]
sizes_from_blocksides(::Type{<:Ridiagonal}, bs::Vector{Int}) = [(bs[i+1],bs[i]) for i=1:length(bs)-1]

# AbstractMatrix methods
# =================================================

size(MR::MiRi)    = (n = sum(block_size(MR,1)); (n,n))
size(MR::MiRi, d) = d::Integer <= 2 ? size(MR)[d] : 1

function getindex(R::Ridiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(block_size(R,1))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if Block4i  == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        wbi == wbj ? one(T) : zero(T)
    elseif  Block4i == Block4j + 1
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        R.data[Block4j][wbi, wbj]
    else 
        return zero(T)
    end
end

function getindex(M::Midiagonal{T}, i::Integer, j::Integer) where T
    row_or_col_Ix = blockedrange(block_size(M,1))
    fbi = findblockindex(row_or_col_Ix, i)
    fbj = findblockindex(row_or_col_Ix, j)
    Block4i = fbi.I[1]
    Block4j = fbj.I[1]
    if Block4i == Block4j
        wbi = blockindex(fbi)
        wbj = blockindex(fbj)
        M.data[Block4i][wbi, wbj]
    else 
        return zero(T)
    end
end


# Convienence (not sure if we want these ?)
# =================================================

function randn(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} 
    M = map(sizes_from_blocksides(A, bs)) do sz
        X = rand(T,sz)
        X * X'
    end
    Midiagonal(M)
end

function randn(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} 
    R = map(sizes_from_blocksides(A, bs)) do sz
        rand(T,sz)
    end
    Ridiagonal(R)    
end 

function eye(::Type{A}, bs::Vector{Int}) where {T,A<:Ridiagonal{T}} 
    R = map(sizes_from_blocksides(A, bs)) do sz
        zeros(T,sz)
    end
    Ridiagonal(R)    
end 

function eye(::Type{A}, bs::Vector{Int}) where {T,A<:Midiagonal{T}} 
    Midiagonal = map(sizes_from_blocksides(A, bs)) do sz
        Matrix(T(1)*I(sz[1]))
    end
    Midiagonal(R)    
end 


# an alternative to these is to simply overload map to reconsititue a Midiagonal?
sqrt(M::Midiagonal)      = Midiagonal(map(sqrt, M.data)) 
Hermitian(M::Midiagonal, uplo=:U) = Midiagonal(map(x->Hermitian(x,uplo), M.data)) 
Symmetric(M::Midiagonal, uplo=:U) = Midiagonal(map(x->Symmetric(x,uplo), M.data)) 
cholesky(M::Midiagonal)  = Midiagonal(map(x->cholesky(x).L, M.data)) 
