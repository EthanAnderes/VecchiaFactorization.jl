using LinearAlgebra
BLAS.set_num_threads(1)
using VecchiaFactorization
import VecchiaFactorization as VF
using BlockArrays
using Random
using Test
using LBblocks


@testset "vecchia_approx.jl and sparse_matrix_show.jl" begin 

    n = 320 

    Σ = @sblock let θtru = 1.5, n
        K = (x,y,θtru) -> exp(- θtru * abs(x - y) ^ 0.8 )
        x = range(0,4,n)
        K.(x, x', θtru)
    end

    block_sizes = [100, 150, 20, 50]
    perm = sortperm(@. sin((1:n)*2*π/n))

    R1, M1, P1 = VF.R_M_P(Σ, block_sizes, perm)
    R2, M2, P2 = VF.R_M_P(Σ[perm, perm], block_sizes)
    R3, M3, P3 = VF.R_M_P((i,j) -> Σ[i,j], block_sizes, perm)

    @test VF.sparse(R1) ≈ VF.sparse(R2) ≈ VF.sparse(R3)
    @test VF.sparse(M1) ≈ VF.sparse(M2) ≈ VF.sparse(M3)
    @test VF.sparse(P1) ≈ VF.sparse(P3)

    @test VF.Matrix(R1) ≈ VF.Matrix(R2) ≈ VF.Matrix(R3)
    @test VF.Matrix(M1) ≈ VF.Matrix(M2) ≈ VF.Matrix(M3)
    @test VF.Matrix(P1) ≈ VF.Matrix(P3)

    @test VF.sparse(R1') ≈ VF.sparse(R1)';
    @test VF.sparse(M1') ≈ VF.sparse(M1)';
    @test VF.sparse(P1') ≈ VF.sparse(P1)';

    VΣ = VF.vecchia(Σ, block_sizes, perm)
    VF.sparse(inv(VΣ))
    VF.sparse(inv(adjoint(VΣ)))
    VF.sparse(adjoint(inv(VΣ)))

    VF.Matrix(VΣ)
    VF.Matrix(inv(VΣ))
    VF.Matrix(adjoint(VΣ))
    VF.Matrix(inv(adjoint(VΣ)))
    VF.Matrix(adjoint(inv(VΣ)))

end


@testset "Ridiagonal and Qidiagonal" begin 

    @sblock let 

        bs = [2, 4, 3]
        @test VF.sizes_from_blocksides(Ridiagonal, bs) == [(4,2), (3,4)]
        @test VF.sizes_from_blocksides(Qidiagonal, bs) == [(2,4), (4,3)]
        @test VF.sizes_from_blocksides(Midiagonal, bs) == [(2,2), (4,4), (3,3)]
        
        Ri = rand(Ridiagonal{Float64}, bs)
        Qi = rand(Qidiagonal{Float64}, bs)
        Mi = rand(Midiagonal{Float64}, bs)

        @test Ri isa Ridiagonal
        @test Qi isa Qidiagonal
        @test Mi isa Midiagonal

        @test VF.diag_block_dlengths(Ri) == [2,4,3]
        @test VF.diag_block_dlengths(Qi) == [2,4,3]
        @test VF.diag_block_dlengths(Mi) == [2,4,3]
        
        @test size(Ri) == (9,9)
        @test size(Qi) == (9,9)
        @test size(Mi) == (9,9)
        
        @test (size(Ri,1), size(Ri,2)) == size(Ri)
        @test (size(Qi,1), size(Qi,2)) == size(Qi)
        @test (size(Mi,1), size(Mi,2)) == size(Mi)

    end


    @sblock let 
        ## bs = vcat(fill(60,7), fill(80,10))
        ## bs = fill(80,20)
        bs = [2, 4, 3, 10]
        ## bs = fill(10,20)
        ## bs = fill(10,200)

        Ri = randn(Ridiagonal{Float64}, bs)
        Qi = randn(Qidiagonal{Float64}, bs)
        Mi = randn(Midiagonal{Float64}, bs)

        v = rand(Float64, sum(bs))

        @inferred Ri * v
        @inferred Ri' * v
        @inferred Ri \ v 
        @test (Ri \ v) ≈ ldiv!(Ri, copy(v)) rtol = 1e-10
        @inferred Qi * v
        @inferred Qi' * v
        @inferred Qi \ v 
        @test (Qi \ v) ≈ ldiv!(Qi, copy(v)) rtol = 1e-10

        ## -----
        x = 1:size(Ri,1)
        v = zeros(Float64, length(x)); 
        for i = 1:2  # it is strange how unstable this inversion is 
            v[rand(x)] = 1
        end
        ## -----
        # 
        ## τ = 10; v = sin.(τ .* 2 .* π .* x ./ x[end])
        ## -----
        ## v = rand(Float64, sum(bs))
        ## -----
        v1 = Ri \ Ri * v
        v2 = Qi \ Qi * v
        w1 = Ri * (Ri \ v)
        w2 = Qi * (Qi \ v)
        @test v ≈ v1 rtol=1e-5
        @test v ≈ v2 rtol=1e-5
        @test v ≈ w1 rtol=1e-5
        @test v ≈ w2 rtol=1e-5

    end

    #=
    using BenchmarkTools

    bs = fill(50,20)        
    Ri = randn(Ridiagonal{Float64}, bs)
    Qi = randn(Qidiagonal{Float64}, bs)
    Mi = randn(Midiagonal{Float64}, bs)
    v  = randn(size(Ri,1))
    w  = randn(size(Ri,1))
    M = randn(size(Ri,1), size(Ri,1))
    @benchmark mul!(w, Mi, v, true, false)  # 9 μs
    @benchmark Mi * v  # 9 μs
    @benchmark Ri * v  # 9 μs
    @benchmark Qi * v  # 9 μs

    @benchmark Mi' * v # 9 μs
    @benchmark Ri' * v # 9 μs
    @benchmark Qi' * v # 9 μs

    @benchmark Mi \ v  # 716.999 μs
    @benchmark Ri \ v  # 9 μs
    @benchmark Qi \ v  # 9 μs

    @benchmark Mi' \ v # 734 μs
    @benchmark Ri' \ v # 9 μs
    @benchmark Qi' \ v # 9 μs

    @benchmark M * v # 112 μs
    =#

end


@testset "Vecchia, InvVecchia, VecchiaPivoted, InvVecchiaPivoted" begin

    K(x,y,θ) = exp(- θ * abs(x - y) ^ 0.8 )
    θtru    = 1.5
    nblocks, blocksz = 6, 150
    bsn    = fill(blocksz, nblocks) 
    n      = sum(bsn)
    x      = vcat(range(0,4,length=n-100), 4*rand(100))
    prm    = sortperm(x)
    Σ      = PseudoBlockArray(K.(x, x', θtru), bsn, bsn)

    V = Vecchia(;
        diag_blocks=[Σ[Block(i,i)] for i = 1:nblocks],
        subdiag_blocks=[Σ[Block(i+1,i)] for i = 1:nblocks-1],
        )
    V′  = inv(V.R) * V.M * inv(V.R)' 
    iV  = inv(V)
    iV′ = inv(V′)

    Vᴾ     = VecchiaPivoted(Vecchia(Σ[prm,prm], bsn), prm)
    Vᴾ′    = Piv(prm)' * inv(Vᴾ.R) * Vᴾ.M * inv(Vᴾ.R)' * Piv(prm)
    iVᴾ    = inv(Vᴾ)
    iVᴾ′   = inv(Vᴾ′)


    matV   = Matrix(V)
    matVᴾ  = Matrix(Vᴾ)
    matiV  = Matrix(iV)
    matiVᴾ = Matrix(iVᴾ)


    v    = randn(size(V,1))

    @test V  * v ≈ matV  * v rtol=1e-5
    @test V  * v ≈ V′    * v rtol=1e-5
    @test Vᴾ * v ≈ matVᴾ * v rtol=1e-5
    @test Vᴾ * v ≈ Vᴾ′   * v rtol=1e-5

    @test iV  * v ≈ matiV  * v rtol=1e-5
    @test iV  * v ≈ iV′    * v rtol=1e-5
    @test iVᴾ * v ≈ matiVᴾ * v rtol=1e-5
    @test iVᴾ * v ≈ iVᴾ′   * v rtol=1e-5


    #=
    using BenchmarkTools

    v  = randn(size(V,1))

    @benchmark V   * v   # 50  μs
    @benchmark V′  * v   # 100 μs
    @benchmark Vᴾ  * v   # 50  μs
    @benchmark Vᴾ′ * v   # 100 μs
    @benchmark iV   * v  # 50  μs
    @benchmark iV′  * v  # 50  μs
    @benchmark iVᴾ  * v  # 50  μs
    @benchmark iVᴾ′ * v  # 50  μs

    R = V.R
    @benchmark R * v          # 15 μs
    @benchmark $(R') * v      # 15 μs
    @benchmark $(inv(R)) * v  # 15 μs
    @benchmark $(inv(R)') * v # 15 μs

    Q = Qidiagonal(map(x->copy(x'), V.R.data))
    @benchmark Q * v          # 15 μs
    @benchmark $(Q') * v      # 15 μs
    @benchmark $(inv(Q)) * v  # 15 μs
    @benchmark $(inv(Q)') * v # 15 ms

    M = V.M
    @benchmark M * v          # 19 μs
    @benchmark $(M') * v      # 20 μs
    @benchmark $(inv(M)) * v  # 20 μs
    @benchmark $(inv(M)') * v # 20 μs
    @benchmark mul!(v, M, $(copy(v))) # 17 μs

    @benchmark Σ * v    # 83 μs
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

    #=
    using PyPlot

    @sblock let V = Vᴾ, Σ = Σ
        mV  = Matrix(V)
        imV = Matrix(inv(V))
        iΣ  = inv(Σ)

        fig,ax = subplots(2,3)
        mV     |> ax[1,1].imshow
        Σ      |> ax[1,2].imshow
        mV - Σ |> ax[1,3].imshow

        imV      .|> abs .|> log |> ax[2,1].imshow
        iΣ       .|> abs .|> log |> ax[2,2].imshow
        imV - iΣ .|> abs .|> log |> ax[2,3].imshow
    end

    @sblock let V
        matRᴴ = VF.Rᴴmat(V)
        matR  = VF.Rmat(V)
        matRᴴ .|> abs .|> log |> matshow; colorbar()
        matR  .|> abs .|> log |> matshow; colorbar()
    end

    =#



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


    ## invcholΣ = inv(cholesky(Hermitian(Σ,:L)).L)
    ## invcholV = VecchiaFactorization.inv_cholesky(V)
    ## invcholVᴾ = VecchiaFactorization.inv_cholesky(Vecchia(Vᴾ.R,Vᴾ.M, Vᴾ.bsds)) # [invperm(Vᴾ.piv), invperm(Vᴾ.piv)]

    #=
    using PyPlot
    v = randn(size(V,1))

    (invcholV \ v)[prm] |> plot
    (invcholVᴾ \ v[prm]) |> plot
    (invcholΣ \ v)[prm] |> plot
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

    #=
    using PyPlot
    fig, ax = subplots(2)
    ax[1].plot(real.(eigen(IVapx).values))
    ax[1].plot(zeros(size(IVapx,1)),":k")
    ax[2].plot(imag.(eigen(IVapx).values))
    ax[2].plot(zeros(size(IVapx,1)),":k")
    =#

end


## @testset "Inv" begin 
## 
##     @test VF.Inv(R') == VF.Inv(R)'
## 
## 
## end




@testset "util.jl" begin 

    @test VF.block_split(20, 10) == [10,10]    
    @test VF.block_split(20, 5) == [5,5,5,5] 
    @test VF.block_split(20, 11) == [11,9]   
    @test VF.block_split(11, 11) == [11]   
    @test VF.block_split(11, 10) == [10,1]   

end

