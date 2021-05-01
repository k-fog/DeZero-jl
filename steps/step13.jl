# step13

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

setcreator!(v::Variable, func::Func) = (v.creator = func)

function backward!(v::Variable)
    if !isdefined(v, :grad)
        v.grad = ones(size(v.data))
    end

    funcs::Vector{Func} = [v.creator]
    while !isempty(funcs)
        f = pop!(funcs)
        gys = [output.grad for output in f.base.outputs]
        gxs = backward(f, gys...)
        isa(gxs, Tuple) || (gxs = (gxs,))

        for (x, gx) in zip(f.base.inputs, gxs)
            x.grad = gx
            isdefined(x, :creator) && push!(funcs, x.creator)
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


# square
mutable struct Square <: Func
    base::FuncBase
    Square() = new(FuncBase())
end

forward(f::Square, x) = x.^2
backward(f::Square, gy) = 2 * f.base.inputs[1].data .* gy
square(x) = Square()(x)


# add
mutable struct Add <: Func
    base::FuncBase
    Add() = new(FuncBase())
end

forward(f::Add, x1, x2) = x1 + x2
backward(f::Add, gy) = gy, gy
add(x1, x2) = Add()(x1, x2)


# main
x = Variable([2.0])
y = Variable([3.0])
z = add(square(x), square(y))
backward!(z)
println(z.data)
println(x.grad)
println(y.grad)