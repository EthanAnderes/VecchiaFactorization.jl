
@testset "util.jl" begin 

    @test VF.block_split(20, 10) == [10,10]    
    @test VF.block_split(20, 5) == [5,5,5,5] 
    @test VF.block_split(20, 11) == [11,9]   
    @test VF.block_split(11, 11) == [11]   
    @test VF.block_split(11, 10) == [10,1]   

end


