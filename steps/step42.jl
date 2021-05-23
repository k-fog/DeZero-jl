# step42

using DeZero
using Random
Random.seed!(0)

const lr = 0.1
const iters = 100

function mean_squared_error(a, b)
    diff = a - b
    return sum(diff^2) / length(diff)
end

function main() 
    x = rand(100, 1)
    y = 5 .+ 2 .* x .+ rand(100, 1)

    W = Variable(zeros(1, 1))
    b = Variable(zeros(1))

    predict(x) = x * W + b

    for i in 1:iters
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(W)
        cleargrad!(b)

        backward!(loss)
        W.data .-= lr * W.grad.data
        b.data .-= lr * b.grad.data
        @show W, b, loss
    end
end

@time main()