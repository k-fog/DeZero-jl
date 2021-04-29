# step06

abstract type Func end

mutable struct Variable
    data::AbstractArray
    grad::AbstractArray
    creator::Func
    Variable(data::AbstractArray) = new(data)
    Variable(data::Nothing) = new()
    Variable() = new()
end

setcreator!(v::Variable, func::Func) = v.creator = func

function backward!(v::Variable)
    if !isdefined(v, :grad)
        v.grad = ones(size(v.data))
    end

    funcs::Vector{Func} = [v.creator]
    while !isempty(funcs)
        f = pop!(funcs)
        x, y = f.input, f.output
        x.grad = backward(f, y.grad)
        if isdefined(x, :creator)
            push!(funcs, x.creator)
        end
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

mutable struct Square <: Func
    input::Variable
    output::Variable
    Square() = new()
end

forward(f::Square, x) = x.^2
backward(f::Square, gy) = 2 .* f.input.data .* gy
square(x) =  Square()(x)


# exp
mutable struct Exp <: Func
    input::Variable
    output::Variable
    Exp() = new()
end

forward(f::Exp, x) = Base.exp.(x)
backward(f::Exp, gy) = Base.exp.(f.input.data) .* gy
exp(x) = Exp()(x)


function numericaldiff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.data .- eps)
    x1 = Variable(x.data .+ eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data .- y0.data) ./ (eps * 2) 
end

x = Variable([0.5])
y = square(exp(square(x)))

backward!(y)
println(x.grad)

Variable(nothing)