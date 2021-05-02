# step27

using DeZero

@create_func Sin
DeZero.forward(f::Sin, x) = Base.sin.(x)
DeZero.backward(f::Sin, gy) = gy .* Base.cos.(f.inputs[1].data)
sin(x::Variable) = Sin()(x)

function my_sin(x, threshold=1e-4)
    y = 0
    for i in 0:100000
        c = (-1)^i / factorial(big(2 * i + 1))
        t = c * x^(2 * i + 1)
        y = y + t
        if all(abs.(t.data) .< threshold) break end
    end
    return y
end 

function main()
    x = Variable([pi / 4])
    y = my_sin(x, 1e-150)
    backward!(y)
    @show y.data
    @show x.grad
    plot_dot_graph(y, false, file="images/sin_1e-150.png")
end

@time main()