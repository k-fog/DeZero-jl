# step 39

using DeZero

function main()
    x = Variable(1:6)
    y = sum(x)
    backward!(y)
    @show y
    @show size(x)
    @show size(x.grad)
end

@time main()