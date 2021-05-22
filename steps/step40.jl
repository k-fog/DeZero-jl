#step 40

using DeZero

function main()
    x = Variable([1, 2, 3])
    @show broadcastto(x, (3, 2))

    x = Variable([1 2 3; 4 5 6;])
   @show sumto(x, (1, 3))

   x1 = Variable([1,2,3])
   x2 = Variable(10)
   y = x1 + x2
   @show y
   backward!(y)
   @show x2.grad
end

@time main()