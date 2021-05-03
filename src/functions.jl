# Sin
@create_func Sin
forward(f::Sin, x) = sin.(x)
backward(f::Sin, gy) = gy * cos(f.inputs[1])
Base.sin(x::Variable) = Sin()(x)

# Cos
@create_func Cos
forward(f::Cos, x) = cos.(x)
backward(f::Cos, gy) = gy * -sin(f.inputs[1])
Base.cos(x::Variable) = Cos()(x)

# Tanh
@create_func Tanh
forward(f::Tanh, x) = Base.tanh.(x)
backward(f::Tanh, gy) = begin
    y = f.outputs[1].value
    return gy * (1 - y * y)
end
Base.tanh(x) = Tanh()(x)

# Reshape
@create_func Reshape shape::Tuple x_shape::Tuple
forward(f::Reshape, x) = begin
    f.x_shape = size(x)
    return reshape(x, f.shape)
end
backward(f::Reshape, gy) = reshape(gy, f.x_shape)
Base.reshape(x::Variable, shape::Tuple) = begin
    if size(x) == shape return asVariable(x) end
    return Reshape(shape, ())(x)
end
Base.reshape(x::Variable, shape::Integer...) = reshape(x, shape)

# Transpose
@create_func Transpose
forward(f::Transpose, x) = transpose(x)
backward(f::Transpose, gy) = transpose(gy)
Base.transpose(x::Variable) = Transpose()(x)

# Adjoint
@create_func Adjoint
forward(f::Adjoint, x) = adjoint(x)
backward(f::Adjoint, gy) = adjoint(gy)
Base.adjoint(x::Variable) = Adjoint()(x)