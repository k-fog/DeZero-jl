module DeZero

export
    # config
    Config,

    # core
    Func,
    Variable,
    isgraddefined,
    cleargrad!,
    asVariable,
    setcreator!,
    backward!,
    @create_func,
    forward,
    backward,
    +, *, -, /, ^,
    sin, cos,

    # utils
    get_dot_graph,
    plot_dot_graph


include("config.jl")
include("core.jl")
include("functions.jl")
include("utils.jl")

end # module DeZero