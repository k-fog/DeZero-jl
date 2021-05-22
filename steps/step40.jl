#step 40

using DeZero

function main()
    x = Variable([1, 2, 3])
    @show broadcastto(x, (3, 2))

    x = Variable([1 2 3; 4 5 6;])
   @show sumto(x, (3, 1))
end

@time main()