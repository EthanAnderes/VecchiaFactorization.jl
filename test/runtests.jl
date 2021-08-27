using LinearAlgebra
BLAS.set_num_threads(1)
using VecchiaFactorization
using BlockArrays
using Test

@testset "VecchiaFactorization.jl" begin
    # Write your tests here.

    K(x,y,θ) = exp(- θ * abs(x - y) ^ 0.8 )

    # data parameters
    θtru    = 1.5
    nblocks, blocksz = 6, 150
    bsn    = fill(blocksz, nblocks) 
    n      = sum(bsn)
    x      = sort(vcat(range(0,4,length=n-100), 4*rand(100)))
    xb     = blocks(PseudoBlockArray(x, bsn))
    Σ₀     = [ K.(xb[i],   xb[i]', θtru) for i = 1:nblocks ]
    Σ₋₁    = [ K.(xb[i+1], xb[i]', θtru) for i = 1:nblocks-1 ]
    R      = [ - (Σ₋₁[i] / Σ₀[i]) for i = 1:nblocks-1 ]
    tailM₀ = [ Σ₀[i+1] + R[i] * Σ₋₁[i]' for i = 1:nblocks-1 ]
    M      = [ Σ₀[1], tailM₀... ]

    # Σ     = BlockArray(K.(x, x', θtru), bsn, bsn)
    # D0    = Matrix.(fill(Eye{Float64}(blocksz),  nblocks))
    # Dm1   = Matrix.(fill(Zeros{Float64}(blocksz,blocksz), nblocks-1))
    # R′    = BlockBidiagonal(D0, Dm1,:L)
    # M′ = BlockDiagonal(Matrix.(fill(Zeros{Float64}(blocksz,blocksz), nblocks)))
    # view(M′, Block(1,1)) .= Σ[Block(1,1)]
    # for i in 2:nblocks
    #     bii  = Block(i,i)
    #     bimi = Block(i,i-1)
    #     A = - Σ[Block(i,i-1)] / Σ[Block(i-1,i-1)]
    #     view(R′, Block(i,i-1))     .= A
    #     view(M′, Block(i,i)) .= Σ[Block(i,i)] + A * Σ[Block(i,i-1)]'
    # end 
    # R = [R′[Block(i,i-1)] for i in 2:nblocks]
    # M = [M′[Block(i,i)] for i in 1:nblocks]

    V      = Vecchia(R, M, bsn)
    iV     = InvVecchia(R, inv.(M), bsn)
    iV′    = inv(V)

    matRᴴ = VecchiaFactorization.Rᴴmat(V)
    matR  = VecchiaFactorization.Rmat(V)

    matV   = Matrix(V)
    imatV  = Matrix(iV)
    imatV′ = Matrix(iV′)
    
    #=
    using PyPlot
    matV    |> matshow; colorbar()
    truV = K.(x, x', θtru) 
    truV  |> matshow; colorbar()
    (matV - truV)  |> matshow; colorbar()
    
    imatV   .|> abs .|> log |> matshow; colorbar()

    matRᴴ   .|> abs .|> log |> matshow; colorbar()
    matR    .|> abs .|> log |> matshow; colorbar()
    =#
    
    v = randn(size(V,1))

    v1 = V * v
    v2 = iV * v

    v1′ = matV * v
    v2′ = imatV * v

    @test v1 ≈ v1′ rtol=1e-5
    @test v2 ≈ v2′ rtol=1e-5

    #=
    abs.(v1 .- v1′) |> plot
    abs.(v2 .- v2′) |> plot
    =# 

    #= 
    using BenchmarkTools
    ## without threads ...
    @benchmark V * v   # 48 μs
    @benchmark iV * v  # 50 μs

    @benchmark $(Matrix(matV)) * v    # 93 μs
    @benchmark $(Matrix(imatV)) * v   # 90 μs

    @benchmark $matV * v    # 108 μs
    @benchmark $imatV * v   # 108 μs

    (@belapsed $(Matrix(matV)) * v) / (@belapsed V * v)     # 1.7
    Base.summarysize((Matrix(matV))) / Base.summarysize(V)  # 3.27
    =#

end
