
"""
`instantiate_inv!(X::Matrix, R::Ridiagonal, M::Midiagonal)`
modifies X to hold the inverse of the corresponding
Vecchia approximation. Note: currently zeros out any non-posdef block of M
"""
function instantiate_inv! end

function instantiate_inv!(X::Matrix{T}, R::Ridiagonal{T}, M::Midiagonal{T}) where {T} 
    blk_sizes = block_size(R,1)
    @assert blk_sizes == block_size(M,1)
    N = length(blk_sizes)

    fill!(X, T(0))
    Σ⁻¹ = BlockedArray(X, blk_sizes, blk_sizes)

    M_ic  = M.data[N] # if M.data[i] isa LowRankCov then this will return a LowRankCov
    Σ⁻¹[Block(N,N)] .= Matrix(pinv(M_ic)) # TODO: check that pinv is the correct method here
    for ic in N-1:-1:1
        M_ic₊1  = M_ic # since we move backwards
        M_ic  = M.data[ic] 
        Σ⁻¹[Block(ic+1,ic)]  .= M_ic₊1 \ R.data[ic]
        Σ⁻¹[Block(ic,ic)]    .= Matrix(pinv(M_ic)) + R.data[ic]' * Σ⁻¹[Block(ic+1,ic)]
        Σ⁻¹[Block(ic, ic+1)] .= Σ⁻¹[Block(ic+1,ic)]'
    end

    return X
end

function instantiate_inv!(X::Matrix{T}, R::Ridiagonal{T}, M::Midiagonal{T}, P::Piv) where {T} 
    instantiate_inv!(X, R, M)
    X .= X[P.perm, P.perm]
    return X
end

function instantiate_inv(R::Ridiagonal{T}, M::Midiagonal{T}) where {T} 
    n = sum(block_size(R,1))
    X = Array{T,2}(undef, n, n)
    return instantiate_inv!(X, R, M)
end

function instantiate_inv(R::Ridiagonal{T}, M::Midiagonal{T}, P::Piv) where {T} 
    n = sum(block_size(R,1))
    X = Array{T,2}(undef, n, n)
    return instantiate_inv!(X, R, M, P)
end




"""
If `C = (A⁻¹+B⁻¹)⁻¹` and both `A` and `B` have vecchia decompsitions 
```
A = inv(R̄)* M̄ * inv(R̄')
B = inv(R̃)* M̃ * inv(R̃')
```
Then  `R, M, M⁻¹ = vecchia_add_inv(R̄, M̄, R̃, M̃)` returns the corresponding 
vecchia decompsitions for C, i.e. 
```
C   = inv(R) * M * inv(R')
C⁻¹ = R' * M⁻¹ * R
```
"""
function vecchia_add_inv(R̄::Ridiagonal{T}, M̄::Midiagonal{T}, R̃::Ridiagonal{T}, M̃::Midiagonal{T}) where {T} 
    # TODO: convert this to LowRankCov methods

    blk_sizes = block_size(M̄,1)
    @assert blk_sizes == block_size(M̃,1)
    N = length(blk_sizes)
    
    Mdata⁻¹ = Vector{Typ_Sym_or_Hrm(T)}(undef, N)
    Rdata   = Vector{Matrix{T}}(undef, N-1)
    
    M̄data⁻¹ = map(x->inv(cholesky(x)), M̄.data)
    M̃data⁻¹ = map(x->inv(cholesky(x)), M̃.data)
    Mdata⁻¹[N] = Sym_or_Hrm(M̄data⁻¹[N] + M̃data⁻¹[N])

    for i=N-1:-1:1
        M̄⁻¹R̄data, M̃⁻¹R̃data = M̄data⁻¹[i+1]*R̄.data[i], M̃data⁻¹[i+1]*R̃.data[i]
        Rdata[i]    = Mdata⁻¹[i+1] \ (M̄⁻¹R̄data + M̃⁻¹R̃data)
        Mdata⁻¹i    = M̄data⁻¹[i] + M̃data⁻¹[i]
        Mdata⁻¹i   += (R̄.data[i] - Rdata[i])' * M̄⁻¹R̄data
        Mdata⁻¹i   += (R̃.data[i] - Rdata[i])' * M̃⁻¹R̃data
        Mdata⁻¹[i] = Sym_or_Hrm(Mdata⁻¹i)
    end

    Mdata = map(x->Sym_or_Hrm(inv(cholesky(x))), Mdata⁻¹)

    return Ridiagonal(Rdata), Midiagonal(Mdata), Midiagonal(Mdata⁻¹)
end

# blocks of lower diag == LB, diagonal blocks == DB
## needs testing
# TODO: convert this to LowRankCov methods
function vecchia_from_inv(Σ⁻¹LB::Vector{TL}, Σ⁻¹DB::Vector{TM}) where {T, TL<:AbstractMatrix{T}, TM<:AbstractMatrix{T}} 

    blk_sizes = block_size(Midiagonal(Σ⁻¹DB),1)
    @assert blk_sizes == block_size(Ridiagonal(Σ⁻¹LB),1)
    N = length(Σ⁻¹DB)
    
    Mdata⁻¹ = Vector{Matrix{T}}(undef, N)
    Rdata   = Vector{Matrix{T}}(undef, N-1)
    
    Mdata⁻¹[N] = Matrix(Σ⁻¹DB[N])

    for i=N-1:-1:1
        Rdata[i]   = Mdata⁻¹[i+1] \ Σ⁻¹LB[i]
        Mdata⁻¹[i] = Σ⁻¹DB[i] -  Rdata[i]'*Σ⁻¹LB[i]
    end

    Mdata = map(x->inv(cholesky(Sym_or_Hrm(x))), Mdata⁻¹)

    return Ridiagonal(Rdata), Midiagonal(Mdata), Midiagonal(Mdata⁻¹)
end




"""
`instantiate_inv_tridiagonal(R::Ridiagonal, M::Midiagonal)` returns a block tridiagonal array (non-Pivoted) which is the
tridiagonal inverse of the corresponding Vecchia approximation.
"""
function instantiate_inv_tridiagonal(R::Ridiagonal{T}, M::Midiagonal{T}) where {T} 
    blk_sizes = block_size(R,1)
    @assert blk_sizes == block_size(M,1)
    N = length(blk_sizes)

    preΣ⁻¹ = instantiate_inv_bidiag_partial(R, M)
    Σ⁻¹    = mortar_Tridiagonal_fill(T(0), blk_sizes)

    for i in 1:N
        Σ⁻¹[Block(i,i)] .= preΣ⁻¹[Block(i,i)]
    end

    for i in 1:N-1
        Σ⁻¹[Block(i+1,i)] .= preΣ⁻¹[Block(i+1,i)]
        Σ⁻¹[Block(i,i+1)] .= Σ⁻¹[Block(i+1,i)]'
    end

    return Σ⁻¹
end

# Low level method that generates just the block diagonals and 
# block lower off diagonals. All the rest is left undefined to save memory. 
# TODO: convert this to LowRankCov methods
function instantiate_inv_bidiag_partial(R::Ridiagonal{T}, M::Midiagonal{T}) where {T} 
    blk_sizes = block_size(R,1)
    @assert blk_sizes == block_size(M,1)
    N    = length(blk_sizes)
    L⁻¹M = map(x->inv(cholesky(Sym_or_Hrm(x)).L), M.data)

    Σ⁻¹ = initalize_bidiag_lblks(T, blk_sizes)

    # Fill in the diagonal and the lower diagonal first
    for ic in N:-1:1 
        if ic == N
            Σ⁻¹[Block(ic,ic)] = L⁻¹M[ic]'*L⁻¹M[ic]
        else 
            C      = L⁻¹M[ic+1] * R.data[ic]
            Σ⁻¹[Block(ic+1,ic)] = L⁻¹M[ic+1]' * C
            Σ⁻¹[Block(ic,ic)]   = L⁻¹M[ic]'*L⁻¹M[ic] + C'*C
        end
    end

    return Σ⁻¹
end


