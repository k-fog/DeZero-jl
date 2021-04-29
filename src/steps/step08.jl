# step06

abstract type Func end

mutable struct Variable
    data::AbstractArray
    grad::AbstractArray
    creator::Func
    Variable(data) = new(data)
end

setcreator!(v::Variable, func::Func) = v.creator = func

function backward!(v::Variable)
    if isdefined(v, :creator)
        f = v.creator
        x = f.input
        x.grad = backward(f, v.grad)
        backward!(x)
    end
end

function (f::Func)(input::Variable)
    x = input.data
    y = forward(f, x)
    output = Variable(y)
    setcreator!(output, f)
    f.input = input
    f.output = output
    return output
end

# square
square(x) =  x.^2
square(x::Variable) = Variable(square(x.data))
∇square(x, gy) = 2 .* x .* gy

mutable struct Square <: Func
    input::Variable
    output::Variable
    Square() = new()
end

forward(f::Square, x) = square(x)
backward(f::Square, gy) = ∇square(f.input.data, gy)


# exp
exp(x::AbstractArray) = Base.exp.(x)
exp(x::Variable) = Variable(Base.exp.(x.data))
∇exp(x, gy) = exp(x) .* gy

mutable struct Exp <: Func
    input::Variable
    output::Variable
    Exp() = new()
end

forward(f::Exp, x) = exp(x)
backward(f::Exp, gy) = ∇exp(f.input.data, gy)


function numericaldiff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.data .- eps)
    x1 = Variable(x.data .+ eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data .- y0.data) ./ (eps * 2) 
end


A, B, C = Square(), Exp(), Square()
x = Variable([0.5])
a = A(x)
b = B(a)
y = C(b)

y.grad = [1.0]
backward!(y)
println(x.grad)