# step18


# Config
module Config
    export setbackprop

    const enable_backprop = Ref(true)
    use_backprop(flag::Bool) = (enable_backprop = flag)
end # Config module

using .Config

function no_grad(f::Function)
    oldvalue = Config.enable_backprop[]
    f()
    Config.use_backprop(oldvalue)
end


# core
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

function backward!(v::Variable; retain_grad=false)
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

        retain_grad || for y in f.base.outputs y.grad = nothing end
    end
end

function (f::Func)(inputs...)
    xs = [x.data for x in inputs]
    ys = forward(f, xs...)
    if !isa(ys, Tuple)
        ys = (ys,)
    end
    outputs = [Variable(y) for y in ys]
    if Config.enable_backprop[]
        f.base.generation = maximum([x.generation for x in inputs])
        for output in outputs
            setcreator!(output, f)
        end
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
    no_grad() do
        x = Variable(ones(100,100,100))
        y = square(square(square(x)))
        backward!(y)
    end
end

@time main()