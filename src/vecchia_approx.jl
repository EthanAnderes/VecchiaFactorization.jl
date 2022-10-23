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
Cholesky of Vecchia Factorization approximation:
```
vecchia_chol(Σ, blk_sizes) -> R⁻¹ cholesky(M)
```
Pivoted Cholesky of Vecchia Factorization approximation:
```
vecchia_chol(Σ, blk_sizes, perm) -> Pᵀ R⁻¹ cholesky(M) P
```
where the argument where `Σ` is an `AbstractMatrix` or a `Function` taking indices `(i,j)` and 
returning `Σ[i,j]`.
"""
function vecchia_chol end 


"""
Sqrt of Vecchia Factorization approximation:
```
vecchia_chol(Σ, blk_sizes) -> R⁻¹ sqrt(M)
```
Pivoted Cholesky of Vecchia Factorization approximation:
```
vecchia_chol(Σ, blk_sizes, perm) -> Pᵀ R⁻¹ sqrt(M) P
```
where the argument where `Σ` is an `AbstractMatrix` or a `Function` taking indices `(i,j)` and 
returning `Σ[i,j]`.
"""
function vecchia_sqrt end 


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





function vecchia(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}; atol=0)
	R, M, P = R_M_P(Σ, blk_sizes, perm; atol)
    # println("Vecchia Factorization approximation: Pᵀ R⁻¹ M R⁻ᴴ P")
	return P' * inv(R) * M * inv(R)' * P
end
function vecchia(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}; atol=0)
	R, M, P = R_M_P(Σ, blk_sizes; atol)
    # println("Vecchia Factorization approximation: R⁻¹ M R⁻ᴴ")
	return inv(R) * M * inv(R)'
end

##

function vecchia_general(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer})
	R, M, P = R_M_P_general(Σ, blk_sizes, perm)
    # println("Vecchia Factorization approximation: Pᵀ R⁻¹ M R⁻ᴴ P")
	return P' * inv(R) * M * inv(R)' * P
end
function vecchia_general(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer})
	R, M, P = R_M_P_general(Σ, blk_sizes)
    # println("Vecchia Factorization approximation: R⁻¹ M R⁻ᴴ")
	return inv(R) * M * inv(R)'
end

##

# function vecchia_chol(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer})
#     R, M, P = R_M_P(Σ, blk_sizes, perm)
#     # preM′ = map(Md -> cholesky(Md).L, M.data)
#     preM′ = map(Md -> cholesky(Sym_or_Hrm(Md,:L)).L, M.data) # add Sym_or_Hrm since we are removing it by default on construction
#     P' * inv(R) * Midiagonal(preM′) * P
# end
# function vecchia_chol(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer})
#     R, M, P = R_M_P(Σ, blk_sizes)
#     # preM′ = map(Md -> cholesky(Md).L, M.data)
#     preM′ = map(Md -> cholesky(Sym_or_Hrm(Md,:L)).L, M.data) # add Sym_or_Hrm since we are removing it by default on construction
#     inv(R) * Midiagonal(preM′)
# end

# ##

# function vecchia_sqrt(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer})
#     R, M, P = R_M_P(Σ, blk_sizes, perm)
#     # preM′ = map(sqrt, M.data)
#     preM′ = map(x->Matrix(sqrt(Sym_or_Hrm(x))), M.data) # add Sym_or_Hrm since we are removing it by default on construction
#     P' * inv(R) * Midiagonal(preM′) * P
# end
# function vecchia_sqrt(Σ::Union{AbstractMatrix, Function}, blk_sizes::AbstractVector{<:Integer})
#     R, M, P = R_M_P(Σ, blk_sizes)
#     # preM′ = map(sqrt, M.data)
#     preM′ = map(x->Matrix(sqrt(Sym_or_Hrm(x))), M.data) # add Sym_or_Hrm since we are removing it by default on construction
#     inv(R) * Midiagonal(preM′)
# end

##

function R_M_P(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes); atol=0) where T
	LinearAlgebra.checksquare(Σ)
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	# M = Vector{Matrix{T}}(undef, N) # default
	M = Vector{LowRankCov{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			# M[ic] = Sym_or_Hrm(Σ[blk_indices[ic], blk_indices[ic]])
			# M[ic] = Σ[blk_indices[ic], blk_indices[ic]] # why does this speed up matrix mult??
			M[ic] = low_rank_cov(Sym_or_Hrm(Σ[blk_indices[ic], blk_indices[ic]],:L);tol=atol) # why does this speed up matrix mult??
		else
			R[ic-1], M[ic] = getR₀M₁₁_lowrankchol(
				Σ[blk_indices[ic-1], blk_indices[ic-1]], 
				Σ[blk_indices[ic], blk_indices[ic-1]], 
				Σ[blk_indices[ic], blk_indices[ic]],
				atol,
			)
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end

function R_M_P(Σfun::Function, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes); atol=0)
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	T = typeof(Σfun(1,2)) # This seems brittle. To do it right, check how `map` does it
	# M = Vector{Typ_Sym_or_Hrm(T)}(undef, N)
	# M = Vector{Matrix{T}}(undef, N)
	M = Vector{LowRankCov{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			# M[ic] = Sym_or_Hrm(Σfun.(blk_indices[ic], blk_indices[ic]'))
			# M[ic] = Σfun.(blk_indices[ic], blk_indices[ic]')  # why does this speed up matrix mult??
			M[ic] = low_rank_cov(Sym_or_Hrm(Σfun.(blk_indices[ic], blk_indices[ic]'),:L);tol=atol)  # why does this speed up matrix mult??
		else
			R[ic-1], M[ic] = getR₀M₁₁_lowrankchol(
				Σfun.(blk_indices[ic-1], blk_indices[ic-1]'), 
				Σfun.(blk_indices[ic], blk_indices[ic-1]'), 
				Σfun.(blk_indices[ic], blk_indices[ic]'),
				atol
			)
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end


####

function R_M_P_general(Σ::AbstractMatrix{T}, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes)) where T
	LinearAlgebra.checksquare(Σ)
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	# M = Vector{Typ_Sym_or_Hrm(T)}(undef, N)
	M = Vector{Matrix{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			# M[ic] = Sym_or_Hrm(Σ[blk_indices[ic], blk_indices[ic]])
			M[ic] = Σ[blk_indices[ic], blk_indices[ic]] # why does this speed up matrix mult??
		else
			R[ic-1], M[ic] = getR₀M₁₁_general(
				Σ[blk_indices[ic-1], blk_indices[ic-1]], 
				Σ[blk_indices[ic], blk_indices[ic-1]], 
				Σ[blk_indices[ic], blk_indices[ic]],
			)
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end

function R_M_P_general(Σfun::Function, blk_sizes::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer}=1:sum(blk_sizes))
	@assert isperm(perm)

	blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
	N = length(blk_sizes)
	T = typeof(Σfun(1,2)) # This seems brittle. To do it right, check how `map` does it
	# M = Vector{Typ_Sym_or_Hrm(T)}(undef, N)
	M = Vector{Matrix{T}}(undef, N)
	R = Vector{Matrix{T}}(undef, N-1)
	for ic in 1:N # loops over the column block index
		if ic == 1
			# M[ic] = Sym_or_Hrm(Σfun.(blk_indices[ic], blk_indices[ic]'))
			M[ic] = Σfun.(blk_indices[ic], blk_indices[ic]')  # why does this speed up matrix mult??
		else
			R[ic-1], M[ic] = getR₀M₁₁_general(
				Σfun.(blk_indices[ic-1], blk_indices[ic-1]'), 
				Σfun.(blk_indices[ic], blk_indices[ic-1]'), 
				Σfun.(blk_indices[ic], blk_indices[ic]'),
			)
		end
	end

    return Ridiagonal(R), Midiagonal(M), Piv(perm)
end

###########

function getR₀M₁₁_posdef(Σ₀₀, Σ₁₀, Σ₁₁, atol)
    U    = force_chol(Sym_or_Hrm(Σ₀₀,:U), atol).U
    C    = Σ₁₀ / U
    R₀   = - C / U'
    M₁₁  = Σ₁₁ - C*C'
    if !isposdef(Sym_or_Hrm(M₁₁))
    	@warn "Non-positive definite Vecchia block detected and was clamped to be positive definite."
    	return R₀, force_posdef(M₁₁, atol)
    else
    	return R₀, M₁₁
    end
end


function getR₀M₁₁_lowrankchol(Σ₀₀, Σ₁₀, Σ₁₁, atol)
    L    = low_rank_chol(Sym_or_Hrm(Σ₀₀,:L); tol=atol)
    C    = Σ₁₀ / L'
    R₀   = - C / L
    M₁₁  = low_rank_cov(Sym_or_Hrm(Σ₁₁ - C*C',:L); tol=atol)

    if !issuccess(M₁₁)
    	@warn "issuccess(M₁₁) == false" maxlog=1
    end

	return R₀, M₁₁
end
# function getR₀M₁₁_lowrankchol(Σ₀₀, Σ₁₀, Σ₁₁, atol)

#     R₀   = - Σ₁₀ / low_rank_cov(Sym_or_Hrm(Σ₀₀,:L);tol=atol)
#     M₁₁  = low_rank_cov(Sym_or_Hrm(Σ₁₁ + R₀ * Σ₁₀',:L);tol=atol)

#     if !issuccess(M₁₁)
#     	@warn "issuccess(M₁₁) == false"
#     end

# 	return R₀, M₁₁
# end
# 
# function getR₀M₁₁_lowrankchol(Σ₀₀, Σ₁₀, Σ₁₁, atol)
#     U    = sqrt(Sym_or_Hrm(Σ₀₀,:L))
#     C    = Σ₁₀ / U
#     R₀   = - C / U'
#     M₁₁  = low_rank_cov(Sym_or_Hrm(Σ₁₁ - C*C',:L); tol=atol)

#     if !issuccess(M₁₁)
#     	@warn "issuccess(M₁₁) == false"
#     end

# 	return R₀, M₁₁
# end


function getR₀M₁₁_general(Σ₀₀, Σ₁₀, Σ₁₁)

    R₀   = - Σ₁₀ / Σ₀₀
    M₁₁  = Σ₁₁ + Sym_or_Hrm(R₀ * Σ₁₀')

    return R₀, M₁₁
end
