using DeZero
using Test

function allclose(a, b; rtol=1e-5, atol=1e-8)
    return all(@. abs(a - b) <= (atol + rtol * abs(b)))
end

function checkgradient(f::Function, x, args...; rtol=1e-4, atol=1e-5)
    x = asvariable(x)
    num_grad = numericalgrad(f, x, args...)
    y = f(x, args...)
    backward!(y)
    bp_grad = x.grad.data
    return allclose(num_grad, bp_grad, rtol=rtol, atol=atol)
end

@testset "BasicMath" begin
    include("basicmath.jl")
end