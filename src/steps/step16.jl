# step16

abstract type Func end

mutable struct Variable
    data::AbstractArray
    grad::Union{AbstractArray,Nothing}
    creator::Func
    generation::Int

    function Variable(data::AbstractArray) 
        v = new(data)
        v.generation = 0
        return v
    end
end

isgraddefined(v::Variable) = isdefined(v, :grad) && !isnothing(v.grad)
cleargrad(v::Variable) = (v.grad = nothing)

mutable struct FuncBase
    inputs::AbstractArray{Variable}
    outputs::AbstractArray{Variable}
    generation::Int
    FuncBase() = new()
end

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.base.generation + 1
end

function backward!(v::Variable)
    if !isgraddefined(v)
        v.grad = ones(size(v.data))
    end

    funcs::Vector{Func} = []
    seen_set = Set()
    addfunc(f) = begin
        if f ∉ seen_set
            push!(funcs, f)
            push!(seen_set, f)
            sort!(funcs, lt=(a, b) -> a.base.generation < b.base.generation)
        end
    end
    addfunc(v.creator)
    while !isempty(funcs)
        f = pop!(funcs)
        gys = [output.grad for output in f.base.outputs]
        gxs = backward(f, gys...)
        isa(gxs, Tuple) || (gxs = (gxs,))

        for (x, gx) in zip(f.base.inputs, gxs)
            x.grad = (isgraddefined(x) ? x.grad : 0) .+ gx
            isdefined(x, :creator) && addfunc(x.creator)
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
    f.base.generation = maximum([x.generation for x in inputs])
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
function main()
    x = Variable([2.0])
    a = square(x)
    y = add(square(a), square(a))
    backward!(y)
    @show y.data
    @show x.grad
end

@time  main()