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
    pow, ^


include("config.jl")
include("core.jl")

end # module DeZero