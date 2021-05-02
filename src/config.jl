module Config
    const enable_backprop = Ref(true)
    use_backprop(flag::Bool) = (enable_backprop[] = flag)
end # module Config

using .Config

function no_grad(f::Function)
    use_backprop(false)
    f()
    use_backprop(true)
end