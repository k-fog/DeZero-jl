# step34

using DeZero
using Plots

function main()
    x = Variable(collect(range(-7, 7, length=200)))
    y = sin(x)
    backward!(y, create_graph=true)
    logs = [vec(y.data)]

    iters = 3
    for i in 1:iters
        push!(logs, vec(x.grad.data))
        gx = x.grad
        cleargrad!(x)
        backward!(gx, create_graph=true)
        @show x.grad.data
    end

    labels = ["y=sin(x)", "y′", "y′′", "y′′′"]
    plt = plot()
    for (i, v) in enumerate(logs)
        plt = plot!(x.data, logs[i], label=labels[i])
    end
    cd("images")
    savefig(plt, "sin")
end

@time main()