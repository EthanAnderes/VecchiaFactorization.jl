
function sparse(X::NTuple{N,InvOrAdjOrVecc}) where {N}
    # only works if there are sparse methods for each Vecchia Factor in X
	foldr(*, map(sparse, X)) 
end

function Matrix(X::NTuple{N,InvOrAdjOrVecc}) where {N}
	foldr(*, map(Matrix, X))
end


# sparse, Matrix and show for Ridiagonal
# =================================================

# ## sparse
function sparse(R::Ridiagonal{T}) where {T}
    n           = size(R,1)
    block_sizes = block_size(R,1)
    Rmat = PseudoBlockArray(
        spdiagm(fill(one(T),n)), 
        block_sizes, 
        block_sizes
    )
    for i=1:length(R.data)
        Rmat[Block(i+1,i)] .= R.data[i] 
    end
    sparse(Rmat)
end

function sparse(adjR::Adjoint{<:Any,<:Ridiagonal})
	adjoint(sparse(adjR.parent))
end

# No sparse representations for invR or inv_adj_R
# sparse(invR::Inv{<:Any,<:Ridiagonal}) 
# sparse(inv_adj_R::Inv{<:Any,<:Adjoint{<:Any,<:Ridiagonal}})

# ## Matrix
Matrix(R::Ridiagonal) = Matrix(sparse(R))

function Matrix(adjR::Adjoint{<:Any,<:Ridiagonal})
	Matrix(adjoint(sparse(adjR.parent)))
end 

function Matrix(invR::Inv{<:Any,<:Ridiagonal})
	inv(Matrix(invR.parent))
end

function Matrix(inv_adj_R::Inv{<:Any,<:Adjoint{<:Any,<:Ridiagonal}})
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

function Base.show(io::IO, m::MIME"text/plain", adjR::Adjoint{<:Any,<:Ridiagonal})
    # show(io, m, sparse(R))    
    println(io, "Adjoint Ridiagonal:")
    X        = sparse(adjR)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end




# sparse, matrix and show for Midiagonal
# =================================================

# ## Sparse
sparse(M::Midiagonal) = sparse(mortar(Diagonal(M.data)))

function sparse(adjM::Adjoint{<:Any,<:Midiagonal})
	adjoint(sparse(adjM.parent))
end

function sparse(invM::Inv{<:Any,<:Midiagonal})
	sparse(Midiagonal(map(inv,invM.parent)))
end

function sparse(inv_adj_R::Inv{<:Any,<:Adjoint{<:Any,<:Midiagonal}})
	sparse(Midiagonal(map(x->inv(x'),invM.parent)))
end


# ## Matrix
Matrix(M::Midiagonal) = Matrix(sparse(M))

Matrix(M::Adjoint{<:Any,<:Midiagonal}) = Matrix(sparse(M))

Matrix(M::Inv{<:Any,<:Adjoint{<:Any,<:Midiagonal}}) = Matrix(sparse(M))

# ## show

function Base.show(io::IO, m::MIME"text/plain", M::Midiagonal)
    println(io, "Midiagonal:")
    X        = sparse(M)
    io       = IOContext(io, :typeinfo => eltype(X))
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end

# sparse, matrix and show for Piv
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




