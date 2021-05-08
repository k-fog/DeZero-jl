@testset "Add" begin
    x1 = Variable([1,2,3])
    x2 = [1,2,3]
    y = x1 + x2
    @test y.data == [2,4,6]
    @test (x2 + x1).data == [2,4,6]
    @test (x1 + x1).data == [2,4,6]
    f = (x, y) -> x + y
    @test checkgradient(f, x1, x2)
    
    x1 = Variable(rand(2,3))
    x2 = Variable(rand(2,3))
    @test checkgradient(f, x1, x2)
end

@testset "Mul" begin
    x1 = Variable([1,2,3])
    x2 = [1,2,3]
    @test (x1 * x2).data == [1,4,9]
    @test (x2 * x1).data == [1,4,9]
    @test (x1 * x1).data == [1,4,9]
end