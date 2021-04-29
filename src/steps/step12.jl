# step12

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

function (f::Func)(inputs...)
    xs = [x.data for x in inputs]
    ys = forward(f, xs...)
    if !isa(ys, Tuple)
        ys = (ys,)
    end
    outputs = [Variable(y) for y in ys]
    for output in outputs
        setcreator!(output, f)
    end
    f.base.inputs = collect(inputs)
    f.base.outputs = outputs
    return length(outputs) > 1 ? outputs : outputs[1]
end

# add
mutable struct Add <: Func
    base::FuncBase
    Add() = new(FuncBase())
end

forward(f::Add, x1, x2) = x1 + x2
add(x1, x2) = Add()(x1, x2)


# main
f = Add()
y = f(Variable([2]), Variable([3]))
println(y.data)