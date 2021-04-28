# step03

mutable struct Variable
    data
    Variable(data) = new(data)
end

square(x) =  x.^2
square(x::Variable) = Variable(square(x.data))

exp(x::Variable) = Variable(Base.exp.(x.data))

A, B, C = square, exp, square
x = Variable([0.5])
a = A(x)
b = B(a)
y = C(b)
println(y.data)