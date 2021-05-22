abstract type Func end

mutable struct Variable{F <: Func}
    data::Array
    creator::F
    grad::Union{Variable,Nothing}
    name::Union{String,Nothing}
    generation::Int

    function Variable(data::AbstractArray, name=nothing) 
        v = new{Func}(data)
        v.name = name
        v.generation = 0
        return v
    end
    Variable(data::Number, name=nothing) = Variable([data], name)
end

Base.show(io::IO, v::Variable) = println(io, "Variable(", v.data, ")")
Base.size(v::Variable) = size(v.data)
Base.length(v::Variable) = length(v.data)
Base.eltype(v::Variable) = Variable
Base.getindex(v::Variable, args...) = getindex(v.data, args...)
function Base.show(io::IO, ::MIME"text/plain", v::Variable{T}) where {T}
    println(io, "Variable{$T}\n", v.data)
end

isdatadefined(v::Variable) = isdefined(v, :data)
isgraddefined(v::Variable) = isdefined(v, :grad) && !isnothing(v.grad)

cleargrad!(v::Variable) = (v.grad = nothing)

asvariable(obj) = obj isa Variable ? obj : Variable(collect(obj))

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.generation + 1
end

function backward!(v::Variable; retain_grad=false, create_graph=false)
    if !isgraddefined(v)
        v.grad = Variable(ones(size(v.data)))
    end

    funcs = Func[]
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
        gys = [output.value.grad for output in f.outputs]

        using_grad(create_graph) do
            gxs = backward(f, gys...)
            gxs isa Tuple || (gxs = (gxs,))

            for (x, gx) in zip(f.inputs, gxs)
                if isgraddefined(x)
                    x.grad = x.grad + gx
                else
                x.grad = gx
                end
                isdefined(x, :creator) && addfunc(x.creator)
            end
        end
        retain_grad || for y in f.outputs y.value.grad = nothing end
    end
end

macro createfunc(name, arg...)
    return quote
        mutable struct $(esc(name)) <: Func
            $(arg...)
            inputs::Array{Variable}
            outputs::Array{WeakRef}
            x_shape::Vector{Tuple}
            generation::Int
            $(esc(name))($(arg...)) = new($(arg...))
        end
    end
end

macro funcfield()
    return quote
        $(esc(:inputs))::Array{Variable}
        $(esc(:outputs))::Array{WeakRef}
        $(esc(:x_shape))::Vector{Tuple}
        $(esc(:generation))::Int
    end
end

function (f::Func)(inputs...)
    inputs = [asvariable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = forward(f, xs...)
    if !(ys isa Tuple)
        ys = (ys,)
    end
    outputs = [Variable(y) for y in ys]
    if Config.enable_backprop[]
        f.generation = maximum([x.generation for x in inputs])
        for output in outputs
            setcreator!(output, f)
        end
    end
    f.inputs = inputs
    f.outputs = [WeakRef(output) for output in outputs]
    return length(outputs) > 1 ? outputs : outputs[1]
end

# Add
@createfunc Add
forward(f::Add, x1, x2) = x1 .+ x2
backward(f::Add, gy) = gy, gy
add(x1, x2) = Add()(x1, x2)
Base.:+(x::Variable, y::Variable) = add(x, y)
Base.:+(x::Variable, y) = add(x, y)
Base.:+(x, y::Variable) = add(x, y)

# Mul
@createfunc Mul
forward(f::Mul, x1, x2) = x1 .* x2
backward(f::Mul, gy) = (gy * f.inputs[2], gy * f.inputs[1])
mul(x1, x2) = Mul()(x1, x2)
Base.:*(x::Variable, y::Variable) = mul(x, y)
Base.:*(x::Variable, y) = mul(x, y)
Base.:*(x, y::Variable) = mul(x, y)

# Neg
@createfunc Neg
forward(f::Neg, x) = -x
backward(f::Neg, gy) = -gy
neg(x) = Neg()(x)
Base.:-(x::Variable) = neg(x)

# Sub
@createfunc Sub
forward(f::Sub, x1, x2) = x1 .- x2
backward(f::Sub, gy) = (gy, -gy)
sub(x1, x2) = Sub()(x1, x2)
Base.:-(x::Variable, y::Variable) = sub(x, y)
Base.:-(x::Variable, y) = sub(x, y)
Base.:-(x, y::Variable) = sub(x, y)

# Div
@createfunc Div
forward(f::Div, x1, x2) = x1 ./ x2
backward(f::Div, gy) = begin
    x1, x2 = f.inputs
    gx1 = gy / x2
gx2 = gy * (-x1 / x2^2)
return gx1, gx2
end
div(x1, x2) = Div()(x1, x2)
Base.:/(x::Variable, y::Variable) = div(x, y)
Base.:/(x::Variable, y) = div(x, y)
Base.:/(x, y::Variable) = div(x, y)

# Pow
@createfunc Pow c::Real
forward(f::Pow, x) = x.^f.c
backward(f::Pow, gy) = f.c * f.inputs[1]^(f.c - 1) * gy
pow(x, c) = Pow(c)(x)
Base.:^(x::Variable, c) = pow(x, c)