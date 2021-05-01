# step02

mutable struct Variable
    data
    Variable(data) = new(data)
end

function square(x)
    return x.^2
end

function square(x::Variable)
    return square(x.data)
end

x = Variable([10.0])
f = square
y = f(x)
println(y)