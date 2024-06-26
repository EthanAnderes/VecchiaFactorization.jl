
# `sparse` and `Matrix` for tuples of Vecchia terms
# =================================================

function sparse(X::NTuple{N,InvOrAdjOrVecc_VF}) where {N}
    # only works if there are sparse methods for each Vecchia Factor in X
	foldr(*, map(sparse, X)) 
end

function Matrix(X::NTuple{N,InvOrAdjOrVecc_VF}) where {N}
	foldr(*, map(Matrix, X))
end


# `sparse`, `Matrix` and `show` for Ridiagonal
# =================================================

# ## sparse
function sparse(R::Ridiagonal{T}) where {T}
    n           = size(R,1)
    block_sizes = block_size(R,1)
    Rmat = BlockedArray(
        spdiagm(fill(one(T),n)), 
        block_sizes, 
        block_sizes
    )
    for i=1:length(R.data)
        Rmat[Block(i+1,i)] .= R.data[i] 
    end
    sparse(Rmat)
end

function sparse(adjR::Adj{<:Any,<:Ridiagonal})
	adjoint(sparse(adjR.parent))
end

# No sparse representations for invR or inv_adj_R
# sparse(invR::Inv{<:Any,<:Ridiagonal}) 
# sparse(inv_adj_R::Inv{<:Any,<:Adj{<:Any,<:Ridiagonal}})

# ## Matrix
function Matrix(R::Ridiagonal{T}) where {T} 
    n           = size(R,1)
    block_sizes = block_size(R,1)
    return mortar(
        Bidiagonal(map(x->zeros(T,x,x) + I, block_sizes), R.data, :L)
    )
end

function Matrix(adjR::Adj{<:Any,<:Ridiagonal})
	adjoint(Matrix(adjR.parent))
end 

function Matrix(invR::Inv{<:Any,<:Ridiagonal})
	inv(Matrix(invR.parent))
end

function Matrix(inv_adj_R::Inv{<:Any,<:Adj{<:Any,<:Ridiagonal}})
	inv(adjoint(Matrix(inv_adj_R.parent.parent)))
end

# ## Show
function Base.show(io::IO, m::MIME"text/plain", R::Ridiagonal)
    # show(io, m, sparse(R))    
    println(io, "Ridiagonal:")
    X        = sparse(R)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end

function Base.show(io::IO, m::MIME"text/plain", adjR::Adj{<:Any,<:Ridiagonal})
    # show(io, m, sparse(R))    
    println(io, "Adj Ridiagonal:")
    X        = sparse(adjR)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end




# `sparse`, `Matrix` and `show` for Midiagonal
# =================================================

# ## Sparse
sparse(M::Midiagonal) = sparse(mortar(Diagonal(Matrix.(M.data))))

function sparse(adjM::Adj{<:Any,<:Midiagonal})
	adjoint(sparse(adjM.parent))
end

function sparse(invM::Inv{<:Any,<:Midiagonal})
	sparse(Midiagonal(map(inv,invM.parent)))
end

function sparse(inv_adj_R::Inv{<:Any,<:Adj{<:Any,<:Midiagonal}})
	sparse(Midiagonal(map(x->inv(x'),inv_adj_R.parent.parent)))
end


# ## Matrix
Matrix(M::Midiagonal) = mortar(Diagonal(Matrix.(M.data)))

Matrix(M::Adj{<:Any,<:Midiagonal}) = adjoint(Matrix(M.parent))

Matrix(M::Inv{<:Any,<:Adj{<:Any,<:Midiagonal}}) = inv(adjoint(Matrix(M.parent.parent)))

# ## show

function Base.show(io::IO, m::MIME"text/plain", M::Midiagonal)
    println(io, "Midiagonal:")
    X        = Matrix(M)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end

# `sparse`, `Matrix` and `show` for Piv
# =================================================

function sparse(p::Piv)
    n = length(p.perm)
    sparse(1:n, p.perm, fill(true,n)) 
end

Matrix(p::Piv) = Matrix(sparse(p))

# Note: we don't need any more methods for adjoint or inv of Piv
# since they always just return another Piv for the return values.

function Base.show(io::IO, m::MIME"text/plain", p::Piv)
    println(io, "Piv permutation type:")
    X        = sparse(p)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end


# generic show for inverse and adjoint
# =================================

function show(io::IO, ::MIME"text/plain", A::InvOrAdj_VF)
    print(io, typeof(A), "\n", A.parent)
end



