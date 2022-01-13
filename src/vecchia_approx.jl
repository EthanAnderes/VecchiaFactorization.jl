# constructor for the vecchia factorization approxiamtion

"""
Vecchia Factorization approximation:
```
vecchia(Σ, blk_sizes) -> R⁻¹ M R⁻ᴴ
```
Pivoted Vecchia Factorization approximation:
```
vecchia(Σ, blk_sizes, perm) -> Pᵀ R⁻¹ M R⁻ᴴ P
```
where the argument where `Σ` is an `AbstractMatrix` or a `Function` taking indices `(i,j)` and 
returning `Σ[i,j]`.
"""
function vecchia end 



"""
Construct individual factors in the Pivoted Vecchia approximation `Pᵀ R⁻¹ M R⁻ᴴ P`.
```
R_M_P(Σ, blk_sizes, perm) -> R, M, P
```
where the argument where `Σ` is an `AbstractMatrix` or a `Function` taking indices `(i,j)` and 
returning `Σ[i,j]`. The return values have the following types:
```
R <: Ridiagonal
M <: Midiagonal
P <: Piv
```
"""
function R_M_P end 





function vecchia(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer})
	R, M, P = R_M_P(Σ, blk_sizes, perm)
    # println("Vecchia Factorization approximation: Pᵀ R⁻¹ M R⁻ᴴ P")
	return P' * inv(R) * M * inv(R)' * P
end

function vecchia(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer})
	R, M, P = R_M_P(Σ, blk_sizes)
    # println("Vecchia Factorization approximation: R⁻¹ M R⁻ᴴ")
	return inv(R) * M * inv(R)'
end

function R_M_P(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes)) where T
	LinearAlgebra.checksquare(Σ)
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	M = Vector{Matrix{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			M[ic] = Σ[blk_indices[ic], blk_indices[ic]]
		else 
			U 		= cholesky(Σ[blk_indices[ic-1], blk_indices[ic-1]]).U 
			C 		= Σ[blk_indices[ic], blk_indices[ic-1]] / U
			R[ic-1] = - C / U'
			M[ic]   = Σ[blk_indices[ic], blk_indices[ic]] - C*C'
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end

# QUESTION: it seems redundant to have two of these R_M_P methods. Perhaps we should just have this one then use dispatch 
# on vecchia(...) above to pass the correct argument
function R_M_P(Σfun::Function, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes))
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	T = typeof(Σfun(1,1)) # This seems brittle. To do it right, check how `map` does it
	M = Vector{Matrix{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			M[ic] = Σfun.(blk_indices[ic], blk_indices[ic]')
		else 
			U 		= cholesky(Σfun.(blk_indices[ic-1], blk_indices[ic-1]')).U 
			C 		= Σfun.(blk_indices[ic], blk_indices[ic-1]') / U
			R[ic-1] = - C / U'
			M[ic]   = Σfun.(blk_indices[ic], blk_indices[ic]') - C*C'
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end





