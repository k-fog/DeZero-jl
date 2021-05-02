# step28

using DeZero
using Plots

rosenbrock(x1,x2) = 100 * (x2 - x1^2)^2 + (x1 - 1)^2

function main()
    x1 = Variable([0.0])
    x2 = Variable([2.0])
    lr = 1e-3
    iters = 10000

    for i in 1:iters
        println(x1.data, x2.data)
        y = rosenbrock(x1, x2)
        cleargrad!(x1)
        cleargrad!(x2)
        backward!(y)
        x1.data -= lr * x1.grad
        x2.data -= lr * x2.grad
    end
end

@time main()