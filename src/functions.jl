@create_func Sin
forward(f::Sin, x) = Base.sin.(x)
backward(f::Sin, gy) = gy * cos(f.inputs[1])
Base.sin(x::Variable) = Sin()(x)

@create_func Cos
forward(f::Cos, x) = Base.cos.(x)
backward(f::Cos, gy) = gy * -sin(f.inputs[1])
Base.cos(x::Variable) = Cos()(x)

@create_func Tanh
forward(f::Tanh, x) = Base.tanh.(x)
backward(f::Tanh, gy) = begin
    y = f.outputs[1].value
    return gy * (1 - y * y)
end
Base.tanh(x) = Tanh()(x)