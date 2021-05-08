# step38

using DeZero

function main()
    x = Variable([1 2 3; 4 5 6])
    @show x'
end

@time main()