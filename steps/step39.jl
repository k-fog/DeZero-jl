# step39

using DeZero

function main()
    x = Variable([1 2 3; 4 5 6;])
    y = sum(x, dims=1)
    backward!(y)
    @show y
    @show x.grad

    x = Variable(randn(2,3,4,5))
    y = sum(x)
    @show y
end

@time main()