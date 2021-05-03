# step35

using DeZero

function main()
    x = Variable([1.0], "x")
    y = tanh(x)
    y.name = "y"
    backward!(y, create_graph=true)

    iters = 6
    for i in 1:(iters - 1)
        println(i)
        gx = x.grad
        cleargrad!(x)
        backward!(gx, create_graph=true)
    end
    gx = x.grad
    gx.name = "gx$(iters)"

    cd("images")
    println("plotting graph...")
    plot_dot_graph(gx, file="tanh$(iters).png")
end

@time main()