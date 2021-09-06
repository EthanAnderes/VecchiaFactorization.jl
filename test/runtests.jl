using LinearAlgebra
BLAS.set_num_threads(1)
using VecchiaFactorization
using BlockArrays
using Random
using Test


@testset "VecchiaFactorization.jl" begin
    # Write your tests here.

    K(x,y,θ) = exp(- θ * abs(x - y) ^ 0.8 )

    # data parameters
    θtru    = 1.5
    nblocks, blocksz = 6, 150
    bsn    = fill(blocksz, nblocks) 
    n      = sum(bsn)
    x    = vcat(range(0,4,length=n-100), 4*rand(100))
    prm  = sortperm(x)

    Σ     = PseudoBlockArray(K.(x, x', θtru), bsn, bsn)
    # V     = Vecchia(Matrix(Σ), bsn)
    V     = Vecchia(;
        diag_blocks=[Σ[Block(i,i)] for i = 1:nblocks],
        subdiag_blocks=[Σ[Block(i+1,i)] for i = 1:nblocks-1],
    )
    Vᴾ    = VecchiaPivoted(Vecchia(Σ[prm,prm], bsn), prm)
    matV  = Matrix(V)
    matVᴾ = Matrix(Vᴾ)

    v    = randn(size(V,1))
    Σv   = Σ * v
    Σv1  = V * v
    Σv2  = matV * v
    Σv3  = Vᴾ * v
    Σv4  = matVᴾ * v

    @test Σv1 ≈ Σv2 rtol=1e-5
    @test Σv3 ≈ Σv4 rtol=1e-5

    #=
    using BenchmarkTools
    v  = randn(size(V,1))
    vv = randn(size(V,1), 500)

    @benchmark V * v    # 49μs
    @benchmark V * vv   # 24ms
    @benchmark Vᴾ * v   # 54 μs
    @benchmark Vᴾ * vv  # 27 ms

    @benchmark Σ * v    # 93 μs
    @benchmark Σ * vv   # 13 ms (this must be faster due to optimized BLAS)
    =#



    #=
    using PyPlot

    fig, ax = subplots(2)
    ax[1].semilogy(eigen(Symmetric(matV)).values,label="matV eigen")
    ax[1].semilogy(eigen(Symmetric(Matrix(Σ))).values,"--",label="Σ eigen")
    ax[1].legend()

    ax[2].semilogy(eigen(Symmetric(matVᴾ)).values,label="matVᴾ eigen")
    ax[2].semilogy(eigen(Symmetric(Matrix(Σ))).values,"--",label="Σ eigen")
    ax[2].legend()
    =# 


    #=
    using PyPlot

    fig, ax = subplots(2)
    ax[1].plot(Σv1,label="Σv1")
    ax[1].plot(Σv ,"--",label="Σv")
    ax[1].legend()

    ax[2].plot(Σv3,label="Σv3")
    ax[2].plot(Σv ,"--",label="Σv")
    ax[2].legend()
    =# 

    iΣ    = inv(Σ)
    iV    = inv(V)
    iVᴾ   = inv(Vᴾ)
    matiV  = Matrix(iV)
    matiVᴾ = Matrix(iVᴾ)

    v   = randn(size(V,1))
    Σv  = Σ * v
    w   = iΣ * Σv
    w1  = iV * Σv
    w2  = matiV * Σv
    w3  = iVᴾ * Σv
    w4  = matiVᴾ * Σv

    @test w1 ≈ w2 rtol=1e-5
    @test w3 ≈ w4 rtol=1e-5


    #=
    using PyPlot

    fig, ax = subplots(2)
    ax[1].plot(w1,label="w1")
    ax[1].plot(w ,"--",label="w")
    ax[1].legend()

    ax[2].plot(w3,label="w3")
    ax[2].plot(w ,"--",label="w")
    ax[2].legend()
    =# 


    invcholΣ = inv(cholesky(Hermitian(Σ,:L)).L)
    invcholV = VecchiaFactorization.inv_cholesky(V)
    invcholVᴾ = VecchiaFactorization.inv_cholesky(Vecchia(Vᴾ.R,Vᴾ.M, Vᴾ.bsds)) # [invperm(Vᴾ.piv), invperm(Vᴾ.piv)]

    #=
    using PyPlot
    v = randn(size(V,1))

    (invcholV \ v)[prm] |> plot
    (invcholVᴾ \ v[prm]) |> plot
    (invcholΣ \ v)[prm] |> plot
    =#



    #=
    using PyPlot

    ## matRᴴ = VecchiaFactorization.Rᴴmat(V)
    ## matR  = VecchiaFactorization.Rmat(V)

    matV    |> matshow; colorbar()
    Σ  |> matshow; colorbar()
    (matV - Σ)  |> matshow; colorbar()
    
    imatV   .|> abs .|> log |> matshow; colorbar()

    matRᴴ   .|> abs .|> log |> matshow; colorbar()
    matR    .|> abs .|> log |> matshow; colorbar()
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



@testset "Vecchia approx inversion" begin

    K(x,y,θ) = exp(- θ * abs(x - y) ^ 0.8 )
    θtru    = 0.2
    nblocks, blocksz = 50, 50
    bsn    = fill(blocksz, nblocks) 
    n      = sum(bsn)
    x      = sort(vcat(range(0,4,length=n-100), 4*rand(100)))

    Σ  = K.(x, x', θtru)
    iΣ = inv(Σ) 
    V  = Vecchia(Σ,bsn) 
    iV = inv(V)  

    IVapx = hcat(map(x->iV*x, eachcol(Σ))...)

    using PyPlot
    fig, ax = subplots(2)
    ax[1].plot(real.(eigen(IVapx).values))
    ax[1].plot(zeros(size(IVapx,1)),":k")
    ax[2].plot(imag.(eigen(IVapx).values))
    ax[2].plot(zeros(size(IVapx,1)),":k")
     
end


