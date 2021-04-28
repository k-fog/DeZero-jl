# step01

mutable struct Variable
    data
    Variable(data) = new(data)
end

data = [1.0]
x = Variable(data)
println(x.data)

x.data = [2.0]
println(x.data)
