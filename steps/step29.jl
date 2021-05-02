# step29

using DeZero

f(x) = x^4 - 2x^2
f′′(x) = @. 12x^2 - 4

function main()
    x = Variable([2.0])
    iters = 10
    for i in 0:iters
        @show i, x.data
        y = f(x)
        cleargrad!(x)
        backward!(y)
        x.data -= x.grad / f′′(x.data)
    end
end

@time main()