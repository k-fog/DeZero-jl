module DeZero

export
    # config
    no_grad,
    use_backprop,

    # core
    Func,
    Variable,
    isgraddefined,
    cleargrad,
    asVariable,
    setcreator!,
    backward!,
    create_func,
    forward,
    backward,
    add, +,
    mul, *,
    neg, -,
    sub, -,
    div, /,
    pow, ^,

    # utils
    get_dot_graph,
    plot_dot_graph


include("config.jl")
include("core.jl")
include("utils.jl")

end # module DeZero