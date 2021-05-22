function numericalgrad(f::Function, x, args...; eps=1e-4)
    x1 = Variable(x.data .- eps)
    x2 = Variable(x.data .+ eps)
    y1 = f(x1, args...)
    y2 = f(x2, args...)
    return (y2.data .- y1.data) / (2 * eps)
end

function _dot_var(v::Variable, verbose=false)
    name = isnothing(v.name) ? "" : v.name
    if verbose && isdatadefined(v)
        if !isnothing(v.name) name *= ": " end
        name *= string(size(v)) * " " * string(eltype(v))
    end
    return "$(objectid(v)) [label=\"$name\", color=orange, style=filled]\n"
end

function _dot_func(f::Func)
    func_name = split(string(typeof(f)), ".")[end]
    txt = "$(objectid(f)) [label=\"$(func_name)\", color=lightblue, style=filled, shape=box]\n"
    for x in f.inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f.outputs
        txt *= "$(objectid(f)) -> $(objectid(y.value))\n"
    end
    return txt
end

function get_dot_graph(output; verbose=true)
    txt = ""
    funcs::Vector{Func} = []
    seen_set = Set()
    addfunc(f) = begin
        if f ∉ seen_set
            push!(funcs, f)
            push!(seen_set, f)
        end
    end
    addfunc(output.creator)
    txt *= _dot_var(output, verbose)

    while !isempty(funcs)
        f = pop!(funcs)
        txt *= _dot_func(f)

        for x in f.inputs
            txt *= _dot_var(x, verbose)
            isdefined(x, :creator) && addfunc(x.creator)
        end
    end
    return "digraph g {\n" * txt * "}"
end

function plot_dot_graph(output; verbose=false, file="graph.png")
    dot_graph = get_dot_graph(output, verbose=verbose)
    graph_path = tempname() * ".dot"
    open(graph_path, "w") do f
        write(f, dot_graph)
    end

    extension = split(file, ".")[end]
    cmd = `dot $graph_path -T $extension -o $file`
    run(cmd)
end