# step33

using DeZero

f(x) = x^4 - 2x^2

function main()
    x = Variable([2.0])
    iters= 10
    for i in 1:iters
        println(i, x.data)
        y = f(x)
        cleargrad!(x)
        backward!(y, create_graph=true)

        gx = x.grad
        cleargrad!(x)
        backward!(gx)
        gx2 = x.grad
        x.data .-= gx.data ./ gx2.data
    end
end

@time main()