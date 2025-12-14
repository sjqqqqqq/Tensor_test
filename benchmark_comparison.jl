using Pkg
Pkg.activate(".")
using Distributed

# Add worker processes if not already added
if nprocs() == 1
    addprocs(4)  # Add 4 worker processes for parallel execution
end

println("="^60)
println("Performance Benchmark: Original vs Optimized (Parallel)")
println("Running on $(nprocs()) processes")
println("="^60)

# Function to run benchmark on a worker process
@everywhere function run_benchmark(script_name)
    t_start = time()
    include(script_name)
    t_end = time()
    elapsed = t_end - t_start
    return (script_name, elapsed)
end

# Run all 4 benchmarks in parallel
println("\nStarting parallel execution of all 4 benchmarks...")
println("="^60)

# Spawn all tasks in parallel
tasks = [
    @spawnat :any run_benchmark("MC_3x100_1st.jl"),
    @spawnat :any run_benchmark("MC_3x100_1st_optimized.jl"),
    @spawnat :any run_benchmark("MC_3x100_2nd.jl"),
    @spawnat :any run_benchmark("MC_3x100_2nd_optimized.jl")
]

# Wait for all tasks to complete and collect results
println("Waiting for all benchmarks to complete...")
results = fetch.(tasks)

# Extract timing results
script_names = [r[1] for r in results]
timings = [r[2] for r in results]

# Map results to variables
t_orig_1st = timings[findfirst(s -> occursin("1st.jl", s) && !occursin("optimized", s), script_names)]
t_opt_1st = timings[findfirst(s -> occursin("1st_optimized", s), script_names)]
t_orig_2nd = timings[findfirst(s -> occursin("2nd.jl", s) && !occursin("optimized", s), script_names)]
t_opt_2nd = timings[findfirst(s -> occursin("2nd_optimized", s), script_names)]

# Display individual results
println("\n" * "="^60)
println("COMPLETED BENCHMARKS")
println("="^60)
for (name, time) in results
    println("$(name): $(round(time, digits=2)) seconds")
end

# Calculate speedups
speedup_1st = t_orig_1st / t_opt_1st
speedup_2nd = t_orig_2nd / t_opt_2nd

# Display comparison results
println("\n" * "="^60)
println("FIRST-ORDER TROTTER COMPARISON")
println("="^60)
println("  Original:  $(round(t_orig_1st, digits=2)) seconds")
println("  Optimized: $(round(t_opt_1st, digits=2)) seconds")
println("  Speedup:   $(round(speedup_1st, digits=2))x")

println("\n" * "="^60)
println("SECOND-ORDER TROTTER COMPARISON")
println("="^60)
println("  Original:  $(round(t_orig_2nd, digits=2)) seconds")
println("  Optimized: $(round(t_opt_2nd, digits=2)) seconds")
println("  Speedup:   $(round(speedup_2nd, digits=2))x")

# Summary
println("\n" * "="^60)
println("OVERALL SUMMARY")
println("="^60)
println("First-order speedup:  $(round(speedup_1st, digits=2))x")
println("Second-order speedup: $(round(speedup_2nd, digits=2))x")
println("="^60)
