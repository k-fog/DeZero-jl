# step04

mutable struct Variable
    data
    Variable(data) = new(data)
end

square(x) =  x.^2
square(x::Variable) = Variable(square(x.data))

exp(x::Variable) = Variable(Base.exp.(x.data))

function numericaldiff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.data .- eps)
    x1 = Variable(x.data .+ eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data .- y0.data) ./ (eps * 2) 
end


f = square
x = Variable([2.0])
dy = numericaldiff(f, x)
println(dy)


function g(x)
    A, B, C = square, exp, square
    return C(B(A(x)))
end

x = Variable([0.5])
@time dy = numericaldiff(g, x)
println(dy)