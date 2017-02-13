using Tensors
using BenchmarkTools
using JLD

include("generate_report.jl")

const SUITE = BenchmarkGroup()

function create_tensors()
    tensor_dict = Dict{Tuple{Int, Int, DataType}, AbstractTensor}()
    symtensor_dict = Dict{Tuple{Int, Int, DataType}, AbstractTensor}()
    for dim in 1:3
        for order in (1,2,4)
            for T in (Float32, Float64)
                tensor_dict[(dim, order, T)] = rand(Tensor{order, dim, T})
                if order != 1
                    symtensor_dict[(dim, order, T)] = rand(SymmetricTensor{order, dim, T})
                else
                    symtensor_dict[(dim, order, T)] = rand(Tensor{order, dim, T})
                end
            end
        end
    end
    return tensor_dict, symtensor_dict
end

tensor_dict, symtensor_dict = create_tensors()

include("benchmark_functions.jl")
include("benchmark_ad.jl")

function run_benchmarks(name, tagfilter = @tagged ALL)
    const paramspath = joinpath(dirname(@__FILE__), "params.jld")
    if !isfile(paramspath)
        println("Tuning benchmarks...")
        tune!(SUITE, verbose=true)
        JLD.save(paramspath, "SUITE", params(SUITE))
    end
    loadparams!(SUITE, JLD.load(paramspath, "SUITE"), :evals, :samples)
    results = run(SUITE[tagfilter], verbose = true, seconds = 2)
    JLD.save(joinpath(dirname(@__FILE__), name * ".jld"), "results", results)
end

function generate_report(v1, v2)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    v2_res = load(joinpath(dirname(@__FILE__), v2 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_$(v1)_$(v2).md"), "w") do f
        printreport(f, judge(minimum(v1_res), minimum(v2_res)); iscomparisonjob = true)
    end
end

function generate_report(v1)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_$(v1).md"), "w") do f
        printreport(f, minimum(v1_res); iscomparisonjob = false)
    end
end

# generate benchmarks for documentation
function generate_docbenchmark()
    at = tensor_dict[(3, 1, Float64)]
    At = tensor_dict[(3, 2, Float64)]
    Ast = symtensor_dict[(3, 2, Float64)]
    AAt = tensor_dict[(3, 4, Float64)]
    AAst = symtensor_dict[(3, 4, Float64)]

    aa = Array(tensor_dict[(3, 1, Float64)])
    Aa = Array(tensor_dict[(3, 2, Float64)])
    Asa = Array(symtensor_dict[(3, 2, Float64)])
    AAa = Array(tensor_dict[(3, 4, Float64)])
    AAsa = Array(symtensor_dict[(3, 4, Float64)])

    open("../docs/src/benchmarks.md", "w") do file
        function printheader(head)
            print(file, "| **$(head)** | | | |\n")
        end
        function printrow(op)
            pretty = (t) -> BenchmarkTools.prettytime(BenchmarkTools.time(minimum(t)))
            speedup = (ta, tt) -> round(10*BenchmarkTools.time(minimum(ta))/BenchmarkTools.time(minimum(tt)))/10
            print(file, "| $(op) | $(pretty(tt)) | $(pretty(ta)) | ×$(speedup(ta, tt)) |\n")
        end

        print(file, """
                    # Benchmarks

                    Here are some benchmark timings for tensors in 3 dimensions. For comparison
                    the timings for the same operations using standard Julia `Array`s are also
                    presented.

                    | Operation  | `Tensor` | `Array` | speed-up |
                    |:-----------|---------:|--------:|---------:|
                    """)

        printheader("Single contraction")
        tt = @benchmark $at ⋅ $at
        ta = @benchmark dot($aa, $aa)
        printrow("a ⋅ a")

        tt = @benchmark $At ⋅ $at
        ta = @benchmark $Aa * $aa
        printrow("A ⋅ a")

        tt = @benchmark $At ⋅ $At
        ta = @benchmark $Aa * $Aa
        printrow("A ⋅ A")

        tt = @benchmark $Ast ⋅ $Ast
        ta = @benchmark $Asa * $Asa
        printrow("As ⋅ As")

        printheader("Double contraction")
        tt = @benchmark $At ⊡ $At
        ta = @benchmark dot($(reshape(Aa, (9,))), $(reshape(Aa, (9,))))
        printrow("A ⊡ A")

        tt = @benchmark $Ast ⊡ $Ast
        ta = @benchmark dot($(reshape(Asa, (9,))), $(reshape(Asa, (9,))))
        printrow("As ⊡ As")

        tt = @benchmark $AAt ⊡ $At
        ta = @benchmark $(reshape(AAa, (9, 9))) * $(reshape(Aa, (9,)))
        printrow("AA ⊡ A")

        tt = @benchmark $AAt ⊡ $AAt
        ta = @benchmark $(reshape(AAa, (9, 9))) * $(reshape(AAa, (9, 9)))
        printrow("AA ⊡ AA")

        tt = @benchmark $AAst ⊡ $AAst
        ta = @benchmark $(reshape(AAsa, (9, 9))) * $(reshape(AAsa, (9, 9)))
        printrow("AAs ⊡ AAs")

        printheader("Outer product")
        tt = @benchmark $at ⊗ $at
        ta = @benchmark $aa * $(aa)'
        printrow("a ⊗ a")

        tt = @benchmark $At ⊗ $At
        ta = @benchmark $(reshape(Aa, (9,))) * $(reshape(Aa, (9,)))'
        printrow("A ⊗ A")

        tt = @benchmark $Ast ⊗ $Ast
        ta = @benchmark $(reshape(Asa, (9,))) * $(reshape(Asa, (9,)))'
        printrow("As ⊗ As")

        printheader("Other operations")
        tt = @benchmark det($At)
        ta = @benchmark det($Aa)
        printrow("det(A)")

        tt = @benchmark det($Ast)
        ta = @benchmark det($Asa)
        printrow("det(As)")

        tt = @benchmark inv($At)
        ta = @benchmark inv($Aa)
        printrow("inv(A)")

        tt = @benchmark inv($Ast)
        ta = @benchmark inv($Asa)
        printrow("inv(As)")

        tt = @benchmark norm($at)
        ta = @benchmark norm($aa)
        printrow("norm(a)")

        tt = @benchmark norm($At)
        ta = @benchmark norm($(Aa[:]))
        printrow("norm(A)")

        tt = @benchmark norm($Ast)
        ta = @benchmark norm($(Asa[:]))
        printrow("norm(As)")

        tt = @benchmark norm($AAt)
        ta = @benchmark norm($(AAa[:]))
        printrow("norm(AA)")

        tt = @benchmark norm($AAst)
        ta = @benchmark norm($(AAsa[:]))
        printrow("norm(AAs)")

        tt = @benchmark $at × $at
        ta = @benchmark $aa × $aa
        printrow("a × a")
    end
end
