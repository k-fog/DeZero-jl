module Config
    export use_backprop

    const enable_backprop = Ref(true)
    use_backprop(flag::Bool) = (enable_backprop = flag)
end # module Config

using .Config

function no_grad(f::Function)
    oldvalue = Config.enable_backprop[]
    f()
    Config.use_backprop(oldvalue)
end