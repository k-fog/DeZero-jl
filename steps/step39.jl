# step 39

using DeZero

function main()
    x = Variable([1 2 3 4; 5 6 7 8])
    y = sum(x, keepdims=true)
    @show ndims(x)
    @show ndims(y)
    backward!(y)
    @show y
    @show size(x)
    @show size(x.grad)
end

@time main()