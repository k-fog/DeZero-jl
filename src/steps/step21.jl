# step21


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

mutable struct Variable{T <: Number}
    data::Array{T}
    grad::Union{Array{T},Nothing}
    name::Union{String,Nothing}
    creator::Func
    generation::Int

    function Variable(data::Array, name=nothing) 
        v = new{eltype(data)}(data)
        v.name = name
        v.generation = 0
        return v
    end
end

isgraddefined(v::Variable) = isdefined(v, :grad) && !isnothing(v.grad)
cleargrad(v::Variable) = (v.grad = nothing)
asVariable(obj) = isa(obj, Variable) ? obj : Variable(collect(obj))

Base.size(v::Variable) = size(v.data)
Base.reshape(v::Variable, dims) = reshape(v.data, dims)
Base.length(v::Variable) = length(v.data)
Base.eltype(v::Variable) = Variable
Base.getindex(v::Variable, args...) = getindex(v.data, args...)
function Base.show(io::IO, ::MIME"text/plain", v::Variable{T}) where {T}
    println(io, "Variable{$T}\n", v.data)
end

macro create_func(name, arg...)
    quote
        mutable struct $(esc(name)) <: Func
            $(arg...)
            inputs::Array{Variable}
            outputs::Array{Variable}
            generation::Int
            $(esc(name))($(arg...)) = new($(arg...))
        end
    end
end

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.generation + 1
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
            sort!(funcs, lt=(a, b) -> a.generation < b.generation)
        end
    end
    addfunc(v.creator)
    while !isempty(funcs)
        f = pop!(funcs)
        gys = [output.grad for output in f.outputs]
        gxs = backward(f, gys...)
        isa(gxs, Tuple) || (gxs = (gxs,))

        for (x, gx) in zip(f.inputs, gxs)
            x.grad = (isgraddefined(x) ? x.grad : 0) .+ gx
            isdefined(x, :creator) && addfunc(x.creator)
        end

        retain_grad || for y in f.outputs y.grad = nothing end
    end
end

function (f::Func)(inputs...)
    inputs = [asVariable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = forward(f, xs...)
    if !isa(ys, Tuple)
        ys = (ys,)
    end
    outputs = [Variable(y) for y in ys]
    if Config.enable_backprop[]
        f.generation = maximum([x.generation for x in inputs])
        for output in outputs
            setcreator!(output, f)
        end
    end
    f.inputs = collect(inputs)
    f.outputs = outputs
    return length(outputs) > 1 ? outputs : outputs[1]
end


# Add
@create_func Add
forward(f::Add, x1, x2) = x1 + x2
backward(f::Add, gy) = gy, gy
add(x1, x2) = Add()(x1, x2)
Base.:+(x::Variable, y) = add(x, y)
Base.:+(x, y::Variable) = add(x, y)

# Mul
@create_func Mul
forward(f::Mul, x1, x2) = x1 .* x2
backward(f::Mul, gy) = (gy .* f.inputs[2].data, gy .* f.inputs[1].data)
mul(x1, x2) = Mul()(x1, x2)
Base.:*(x::Variable, y) = mul(x, y)
Base.:*(x, y::Variable) = mul(x, y)

# Neg
@create_func Neg
forward(f::Neg, x) = -x
backward(f::Neg, gy) = -gy
neg(x) = Neg()(x)
Base.:-(x::Variable) = neg(x)

# Sub
@create_func Sub
forward(f::Sub, x1, x2) = x1 .- x2
backward(f::Sub, gy) = (gy, -gy)
sub(x1, x2) = Sub()(x1, x2)
Base.:-(x::Variable, y) = sub(x, y)
Base.:-(x, y::Variable) = sub(x, y)

# Div
@create_func Div
forward(f::Div, x1, x2) = x1 ./ x2
backward(f::Div, gy) = begin
    x1, x2 = f.inputs[1].data, f.inputs[2].data
    gx1 = gy ./ x2
    gx2 = gy .* (-x1 ./ x2 ^ 2)
    return gx1, gx2
end
div(x1, x2) = Div()(x1, x2)
Base.:/(x::Variable, y) = div(x, y)
Base.:/(x, y::Variable) = div(x, y)

# Pow
@create_func Pow c
forward(f::Pow, x) = x .^ f.c
backward(f::Pow, gy) = @. f.c * f.inputs[0].data ^ (f.c - 1) * gy
pow(x, c) = Pow(c)(x)
Base.:^(x::Variable, c) = pow(x, c)


# main
function main()
    x = Variable([6.0])

    y = x + 2
    @show y

    y = x * 2
    @show y

    y = x / 2
    @show y

    y = -x
    @show y

    y = x - 2
    @show y

    y = x ^ 2
    @show y
end

@time main()