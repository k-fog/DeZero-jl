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
Base.reshape(x::Variable, shape...) = begin
    if size(x) == shape return x end
    if length(shape) == 1 && shape[1] isa Union{Tuple,Array}
        shape = shape[1]
    end
    return Reshape(tuple(shape...))(x)
        end

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
@createfunc Sum axis::Union{Int,Tuple,Nothing} keepdims::Bool
forward(f::Sum, x) = begin
f.x_shape = size(x)
    if f.axis isa Nothing
        y = sum(x)
    else
        y = sum(x, dims=f.axis)
end
    if f.keepdims
        x_dims = length(f.x_shape)
        y_dims = ndims(y)
        if y isa Number y = [y] end
        y = reshape(y, (size(y)..., ones(Int, x_dims - y_dims - 1)...))
    end
    return y
end
backward(f::Sum, gy) =  broadcastto(gy, f.x_shape)
Base.sum(x::Variable; axis=nothing, keepdims=false) = Sum(axis, keepdims)(x)
    
# BroadcastTo
@createfunc BroadcastTo shape::Tuple
forward(f::BroadcastTo, x) = begin
    f.x_shape = size(x)
    return x .+ zeros(f.shape)
end
backward(f::BroadcastTo, gy) = sumto(gy, f.x_shape)
broadcastto(x::Variable, shape) = if size(x) == shape asvariable(x) else BroadcastTo(shape)(x) end

# SumTo
@createfunc SumTo shape::Tuple
forward(f::SumTo, x) = begin
    ndim = length(f.shape)
    lead = ndims(x) - ndim
    lead_axis = tuple(0:lead...)
    axis = tuple([i + lead for (i, sx) in enumerate(f.shape) if sx == 1]...)
    y = sum(x, dims=lead_axis .+ axis)
    lead > 0 && squeeze!(y, lead_axis)
    return y
end
backward(f::SumTo, gy) = broadcastto(gy, f.x_shape)
sumto(x::Variable, shape) = if size(x) == shape x else SumTo(shape)(x) end

# MatNul
@createfunc MatMul
forward(f::MatMul, W, x) = W * x
backward(f::MatMul, gy) = begin
    W, x = f.inputs
    gW = matmul(gy, transpose(x))
    gx = matmul(transpose(W), gy)
    return gW, gx
end
matmul(W, x) = MatMul()(W, x)
Base.:*(A::Variable, B::Variable) = matmul(A, B)
Base.:*(A::Variable, B) = matmul(A, B)
Base.:*(A, B::Variable) = matmul(A, B)