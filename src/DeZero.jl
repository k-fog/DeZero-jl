module DeZero

export
    # config
    Config,

    # core
    Func,
    Variable,
    isgraddefined,
    cleargrad!,
    asvariable,
    setcreator!,
    backward!,
    @create_func,
    forward,
    backward,
    add, +, 
    mul, *, 
    neg, sub, -,
    div, /, 
    pow, ^,
    sin, cos,
    transpose, adjoint,

    # utils
    numericalgrad,
    get_dot_graph,
    plot_dot_graph


include("config.jl")
include("core.jl")
include("functions.jl")
include("utils.jl")

end # module DeZero