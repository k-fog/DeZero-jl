# step41

using DeZero

function main()
    W = Variable(randn(2, 3))
    x = Variable(randn(3, 4))
    y = W * x
    backward!(y)
    @show size(W.grad)
    @show size(x.grad)
end

@time main()