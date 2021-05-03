# step37

using DeZero

function main()
    x = Variable([1 2 3; 4 5 6])
    c = Variable([10 20 30; 40 50 60])
    t = x + c
    @show size(t)
    @show sum(t.data)
end

@time main()