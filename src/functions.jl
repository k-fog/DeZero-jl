@create_func Sin
forward(f::Sin, x) = Base.sin.(x)
backward(f::Sin, gy) = gy * cos(f.inputs[1])
Base.sin(x::Variable) = Sin()(x)

@create_func Cos
forward(f::Cos, x) = Base.cos.(x)
backward(f::Cos, gy) = gy * -sin(f.inputs[1])
Base.cos(x::Variable) = Cos()(x)