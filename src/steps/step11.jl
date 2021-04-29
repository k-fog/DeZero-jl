# step11
using Test

abstract type Func end

mutable struct Variable
    data::AbstractArray
    grad::AbstractArray
    creator::Func
    Variable(data::AbstractArray) = new(data)
    Variable(data::Nothing) = new()
    Variable() = new()
end

mutable struct FuncBase
    inputs::AbstractArray{Variable}
    outputs::AbstractArray{Variable}
    FuncBase() = new()
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

function (f::Func)(inputs)
    xs = [x.data for x in inputs]
    ys = forward(f, xs)
    outputs = [Variable(y) for y in ys]
    for output in outputs
        setcreator!(output, f)
    end
    f.base.inputs = inputs
    f.base.outputs = outputs
    return outputs
end

# add
mutable struct Add <: Func
    base::FuncBase
    Add() = new(FuncBase())
end

forward(f::Add, xs) = (xs[1] + xs[2],)


# main
xs = [Variable([2.0]), Variable([3.0])]
f = Add()
ys = f(xs)
y = ys[1]
println(y.data)