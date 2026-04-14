# HEngg_ode.jl
# Time evolution from input pulse (JLD2 file) - ODE simulation with visualization
# Both H_hop and H_int are on for all time (no alternating)
# Usage: julia --project=. HEngg_ode.jl [pulse_file.jld2]

using LinearAlgebra
using DifferentialEquations
using Plots
using JLD2

# ===== QUANTUM SYSTEM SETUP =====
const Gamma = Float64[0 1 1 0;
                      1 0 0 1;
                      1 0 0 1;
                      0 1 1 0]

const I4 = Matrix{ComplexF64}(I, 4, 4)

function projector(site_idx)
    e = zeros(ComplexF64, 4, 1)
    e[site_idx + 1] = 1.0
    return e * e'
end

const P_a_sites = [kron(projector(s), I4) for s in 0:3]
const P_b_sites = [kron(I4, projector(t)) for t in 0:3]
const Hint_base = sum(kron(projector(s), projector(s)) for s in 0:3)
const Hhop_a = kron(Gamma, I4)
const Hhop_b = kron(I4, Gamma)

function idx(s, t)
    return s * 4 + t + 1
end

# ===== PULSE STRUCT =====
mutable struct PulseFull
    n::Int
    T::Float64
    dt::Float64
    Va::Matrix{Float64}
    Vb::Matrix{Float64}
    U::Vector{Float64}
    Ja::Vector{Float64}
    Jb::Vector{Float64}

    function PulseFull(pulse_file::String)
        if !isfile(pulse_file)
            error("Pulse file not found: $pulse_file")
        end

        data = load(pulse_file)
        n  = data["n"]
        T  = data["T"]
        dt = data["dt"]
        U  = data["U"]
        Ja = data["Ja"]
        Jb = data["Jb"]

        # Support two on-disk layouts:
        # (A) Va/Vb as (n×4) absolute-potential matrices  [GRAPE_pulse, optimized_pulse]
        # (B) Va1/Va2/Va3, Vb1/Vb2/Vb3 as relative-potential vectors  [GRAPE_2d_pulse]
        #     Convention: H_Va1 = P[site1]-P[site0], etc.
        #     → absolute: site0 = -(Va1+Va2+Va3), site1 = Va1, site2 = Va2, site3 = Va3
        if haskey(data, "Va")
            Va = data["Va"]
            Vb = data["Vb"]
        else
            Va1, Va2, Va3 = data["Va1"], data["Va2"], data["Va3"]
            Vb1, Vb2, Vb3 = data["Vb1"], data["Vb2"], data["Vb3"]
            Va = hcat(-(Va1 .+ Va2 .+ Va3), Va1, Va2, Va3)
            Vb = hcat(-(Vb1 .+ Vb2 .+ Vb3), Vb1, Vb2, Vb3)
        end

        return new(n, T, dt, Va, Vb, U, Ja, Jb)
    end
end

# Build Hamiltonian at step k (1-indexed) with both H_int and H_hop on
function H_step(p::PulseFull, k::Int)
    H = zeros(ComplexF64, 16, 16)

    # Site potentials
    for s in 1:4
        H .+= p.Va[k, s] .* P_a_sites[s]
    end
    for t in 1:4
        H .+= p.Vb[k, t] .* P_b_sites[t]
    end

    # Interaction term
    H .+= p.U[k] .* Hint_base

    # Hopping terms (both always on)
    H .+= p.Ja[k] .* Hhop_a
    H .+= p.Jb[k] .* Hhop_b

    return H
end

# ===== TIME EVOLUTION FUNCTIONS =====

function H_at_time(t::Float64, p::PulseFull)
    # Each step has duration dt, total time T = n * dt
    k = floor(Int, t / p.dt) + 1
    if k > p.n
        k = p.n
    end
    return H_step(p, k)
end

function schrodinger!(dpsi, psi, params, t)
    p = params
    H = H_at_time(t, p)
    dpsi .= -1im * H * psi
end

function evolve_ode(pulse::PulseFull, psi0::Vector{ComplexF64};
                    n_points::Int=500, abstol::Float64=1e-10, reltol::Float64=1e-10)
    tspan = (0.0, pulse.T)
    saveat = range(0.0, pulse.T, length=n_points)

    prob = ODEProblem(schrodinger!, psi0, tspan, pulse)
    sol = solve(prob, Tsit5(), saveat=saveat, abstol=abstol, reltol=reltol)

    return collect(sol.t), sol.u
end

# ===== PULSE INTERPOLATION FOR PLOTTING =====
function get_pulse_values_at_times(pulse::PulseFull, times::Vector{Float64})
    n_times = length(times)

    # V differences relative to site 0
    V12_vals = zeros(n_times)
    V21_vals = zeros(n_times)
    V22_vals = zeros(n_times)
    U_vals = zeros(n_times)
    Ja_vals = zeros(n_times)
    Jb_vals = zeros(n_times)

    for (i, t) in enumerate(times)
        k = floor(Int, t / pulse.dt) + 1
        if k > pulse.n
            k = pulse.n
        end

        # All parameters active at each time step
        V12_vals[i] = pulse.Va[k, 2] - pulse.Va[k, 1]
        V21_vals[i] = pulse.Va[k, 3] - pulse.Va[k, 1]
        V22_vals[i] = pulse.Va[k, 4] - pulse.Va[k, 1]
        U_vals[i] = pulse.U[k]
        Ja_vals[i] = pulse.Ja[k]
        Jb_vals[i] = pulse.Jb[k]
    end

    return V12_vals, V21_vals, V22_vals, U_vals, Ja_vals, Jb_vals
end

## ===== MAIN EXECUTION =====

# Initial state: |0,1⟩
psi0 = zeros(ComplexF64, 16)
psi0[idx(0, 1)] = 1.0

# Target state: (|0,1⟩ + |2,3⟩)/sqrt(2)
psi_target = zeros(ComplexF64, 16)
psi_target[idx(0, 1)] = 1.0 / sqrt(2)
psi_target[idx(2, 3)] = 1.0 / sqrt(2)

# Collect all pulse files (command-line or default set)
pulse_files = length(ARGS) >= 1 ? ARGS : ["GRAPE_pulse.jld2", "optimized_pulse.jld2", "GRAPE_2d_pulse.jld2"]

# ── per-file simulation ─────────────────────────────────────────────────────
all_times  = Dict{String, Vector{Float64}}()
all_states = Dict{String, Vector{Vector{ComplexF64}}}()

println("="^60)
for pulse_file in pulse_files
    println("Loading pulse from: $pulse_file")
    pulse = PulseFull(pulse_file)
    println("  n=$(pulse.n), T=$(pulse.T), dt=$(pulse.dt)")

    times_i, states_i = evolve_ode(pulse, psi0, n_points=500)
    all_times[pulse_file]  = times_i
    all_states[pulse_file] = states_i

    fid_final = abs2(dot(psi_target, states_i[end]))
    println("  Final fidelity (ODE): $fid_final")
    println("-"^60)
end

# ── comparison summary ──────────────────────────────────────────────────────
println("\nFinal-state comparison  (target: (|0,1⟩ + |2,3⟩)/√2)")
println("="^60)
ref_file = pulse_files[1]
ref_state = all_states[ref_file][end]
for pulse_file in pulse_files
    psi_f = all_states[pulse_file][end]
    fid   = abs2(dot(psi_target, psi_f))
    overlap_with_ref = abs2(dot(ref_state, psi_f))
    println("$pulse_file")
    println("  |⟨target|ψ_f⟩|²          = $(round(fid, digits=8))")
    println("  |⟨ψ_f[ref]|ψ_f⟩|²        = $(round(overlap_with_ref, digits=8))")
    println("  |ψ[0,1]|² = $(round(abs2(psi_f[idx(0,1)]), digits=6))  " *
            "|ψ[2,3]|² = $(round(abs2(psi_f[idx(2,3)]), digits=6))")
    println()
end

# For plotting, use the last file specified (or first by default)
pulse_file = pulse_files[end]
pulse      = PulseFull(pulse_file)
times      = all_times[pulse_file]
states     = all_states[pulse_file]
fidelity   = [abs2(dot(psi_target, psi)) for psi in states]

# Get pulse parameter values at each time point
V12_vals, V21_vals, V22_vals, U_vals, Ja_vals, Jb_vals = get_pulse_values_at_times(pulse, times)

println("(Plots shown for: $pulse_file)")

# ===== PLOTTING =====
println("Generating plots...")

# Set plot defaults
default(fontfamily="Computer Modern", titlefontsize=14, guidefontsize=12,
        tickfontsize=10, legendfontsize=10, linewidth=2)

# Figure 1: Fidelity vs Time
p1 = plot(times, fidelity,
            xlabel="Time", ylabel="Fidelity",
            title="Fidelity vs Time",
            label="F(t)", color=:blue, legend=:bottomright)
hline!([1.0], linestyle=:dash, color=:gray, label="Target", alpha=0.5)

# Figure 2: V12, V21, V22 vs Time
p2 = plot(times, V12_vals, label="V12 (Vb site 1)", color=:red)
plot!(times, V21_vals, label="V21 (Va site 2)", color=:green)
plot!(times, V22_vals, label="V22 (Vb site 2)", color=:blue)
plot!(xlabel="Time", ylabel="Potential",
        title="Site Potentials vs Time", legend=:topright)

# Figure 3: U vs Time
p3 = plot(times, U_vals,
            xlabel="Time", ylabel="U",
            title="Interaction Strength U vs Time",
            label="U(t)", color=:purple)

# Figure 4: Ja, Jb vs Time
p4 = plot(times, Ja_vals, label="Ja(t)", color=:orange)
plot!(times, Jb_vals, label="Jb(t)", color=:cyan)
plot!(xlabel="Time", ylabel="Hopping Strength",
        title="Hopping Parameters vs Time", legend=:topright)

# Combine all plots into a 2x2 layout
combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))

# Save plots
# savefig(p1, "fidelity_vs_time.png")
# savefig(p2, "V_vs_time.png")
# savefig(p3, "U_vs_time.png")
# savefig(p4, "J_vs_time.png")
# savefig(combined_plot, "ode_simulation_results.png")

# println("\nPlots saved:")
# println("  1. fidelity_vs_time.png")
# println("  2. V_vs_time.png")
# println("  3. U_vs_time.png")
# println("  4. J_vs_time.png")
# println("  5. ode_simulation_results.png (combined)")

