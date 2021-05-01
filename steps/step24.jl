# step24

using DeZero

function test(f)
    x = Variable([1.0])
    y = Variable([1.0])
    z = f(x, y)
    backward!(z)
    @show f
    @show x.grad
    @show y.grad
end

sphere(x, y) = x^2 + y^2
matyas(x, y) = 0.26 * (x^2 + y^2) - 0.48  * x * y
goldstein(x, y) = (1 + (x + y + 1)^2 * (19 - 14 * x + 3 * x^2 - 14 * y + 6 * x * y + 3 * y^2)) *
    (30 + (2 * x - 3 * y)^2 * (18 - 32 * x + 12 * x^2 + 48 * y - 36 * x * y + 27 * y^2))

function main()
    test(sphere)
    test(matyas)
    test(goldstein)
end

@time main()