module Config
    const enable_backprop = Ref(true)
    use_backprop(flag::Bool) = (enable_backprop[] = flag)
end # module Config

using .Config

function using_grad(f::Function, use::Bool)
    oldvalue = Config.enable_backprop[]
    Config.use_backprop(use)
    f()
    Config.use_backprop(oldvalue)
end

nograd(f::Function) = using_grad(f, false)