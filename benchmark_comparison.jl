using Pkg
Pkg.activate(".")

println("="^60)
println("Performance Benchmark: Original vs Optimized")
println("="^60)

# Test with smaller parameters for quick comparison
function run_benchmark(script_name)
    println("\nRunning: $script_name")
    t_start = time()
    include(script_name)
    t_end = time()
    elapsed = t_end - t_start
    println("Completed in: $(round(elapsed, digits=2)) seconds")
    return elapsed
end

# Benchmark first-order Trotter
println("\n" * "="^60)
println("FIRST-ORDER TROTTER COMPARISON")
println("="^60)

println("\n1. Running ORIGINAL version...")
t_orig_1st = run_benchmark("MC_3x100_1st.jl")

println("\n2. Running OPTIMIZED version...")
t_opt_1st = run_benchmark("MC_3x100_1st_optimized.jl")

speedup_1st = t_orig_1st / t_opt_1st
println("\n" * "-"^60)
println("First-order Results:")
println("  Original:  $(round(t_orig_1st, digits=2)) seconds")
println("  Optimized: $(round(t_opt_1st, digits=2)) seconds")
println("  Speedup:   $(round(speedup_1st, digits=2))x")
println("-"^60)

# Benchmark second-order Trotter
println("\n" * "="^60)
println("SECOND-ORDER TROTTER COMPARISON")
println("="^60)

println("\n1. Running ORIGINAL version...")
t_orig_2nd = run_benchmark("MC_3x100_2nd.jl")

println("\n2. Running OPTIMIZED version...")
t_opt_2nd = run_benchmark("MC_3x100_2nd_optimized.jl")

speedup_2nd = t_orig_2nd / t_opt_2nd
println("\n" * "-"^60)
println("Second-order Results:")
println("  Original:  $(round(t_orig_2nd, digits=2)) seconds")
println("  Optimized: $(round(t_opt_2nd, digits=2)) seconds")
println("  Speedup:   $(round(speedup_2nd, digits=2))x")
println("-"^60)

# Summary
println("\n" * "="^60)
println("OVERALL SUMMARY")
println("="^60)
println("First-order speedup:  $(round(speedup_1st, digits=2))x")
println("Second-order speedup: $(round(speedup_2nd, digits=2))x")
println("="^60)
