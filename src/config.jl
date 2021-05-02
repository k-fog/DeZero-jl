module Config
    const enable_backprop = Ref(true)
    use_backprop(flag::Bool) = (enable_backprop[] = flag)

    const variable_type = Ref(Float64)
    set_variable_type(t::Type) = (variable_type[] = t)
end # module Config

using .Config

function no_grad(f::Function)
    use_backprop(false)
    f()
    use_backprop(true)
end