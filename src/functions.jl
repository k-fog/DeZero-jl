# Sin
@createfunc Sin
forward(f::Sin, x) = sin.(x)
backward(f::Sin, gy) = gy * cos(f.inputs[1])
Base.sin(x::Variable) = Sin()(x)

# Cos
@createfunc Cos
forward(f::Cos, x) = cos.(x)
backward(f::Cos, gy) = gy * -sin(f.inputs[1])
Base.cos(x::Variable) = Cos()(x)

# Tanh
@createfunc Tanh
forward(f::Tanh, x) = Base.tanh.(x)
backward(f::Tanh, gy) = begin
    y = f.outputs[1].value
    return gy * (1 - y * y)
end
Base.tanh(x) = Tanh()(x)

# Reshape
@createfunc Reshape shape::Tuple
forward(f::Reshape, x) = begin
    f.x_shape = size(x)
    return reshape(x, f.shape)
end
backward(f::Reshape, gy) = reshape(gy, f.x_shape)
Base.reshape(x::Variable, shape::Tuple) = begin
    if size(x) == shape return asvariable(x) end
    return Reshape(shape)(x)
end
Base.reshape(x::Variable, shape::Integer...) = reshape(x, shape)

# Transpose
@createfunc Transpose
forward(f::Transpose, x) = transpose(x)
backward(f::Transpose, gy) = transpose(gy)
Base.transpose(x::Variable) = Transpose()(x)

# Adjoint
@createfunc Adjoint
forward(f::Adjoint, x) = adjoint(x)
backward(f::Adjoint, gy) = adjoint(gy) # ???
Base.adjoint(x::Variable) = Adjoint()(x)

# Sum
@createfunc Sum dims::Int keepdims::Bool
forward(f::Sum, x) = begin
    f.x_shape = [size(x)]
    return sum(x, dims=f.dims)
end
backward(f::Sum, gy) = begin
    return gy * ones(f.x_shape[1])
end
Base.sum(x::Variable; dims=1, keepdims=false) = Sum(dims, keepdims)(x)

# BroadcastTo
@createfunc BroadcastTo shape::Tuple
forward(f::BroadcastTo, x) = begin
    f.x_shape = [size(x)]
    fill(x, f.shape)
end
backward(f::BroadcastTo, gy) = sumto(gy, f.x_shape)
broadcastto(x::Variable, shape) = if size(x) == shape asvariable(x) else BroadcastTo(shape)(x) end

# SumTo
@createfunc SumTo shape::Tuple
forward(f::SumTo, x) = begin
    f.x_shape = [size(x)]
    return sum(x, dims=2)
end
backward(f::SumTo, gy) = broadcastto(gy, f.x_shape)
sumto(x::Variable, shape) = if size(x) == shape asvariable(x) else SumTo(shape)(x) end