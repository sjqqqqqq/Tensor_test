# GRAPE_test.jl
# Compute optimal pulse using GRAPE (via QuantumControl.jl) for the Hubbard-like system
# Uses Hamiltonian, initial/target states from HEngg_ode.jl
# Saves the optimized pulse to GRAPE_pulse.jld2

using LinearAlgebra
using QuantumControl
using QuantumPropagators: ExpProp
using GRAPE
using JLD2

# ===== QUANTUM SYSTEM SETUP (from HEngg_ode.jl) =====
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
const Hint_base = Matrix{ComplexF64}(sum(kron(projector(s), projector(s)) for s in 0:3))
const Hhop_a = Matrix{ComplexF64}(kron(Gamma, I4))
const Hhop_b = Matrix{ComplexF64}(kron(I4, Gamma))

function idx(s, t)
    return s * 4 + t + 1
end

# ===== GRAPE OPTIMIZATION PARAMETERS =====
n_steps = 100           # Number of time steps
T = 2π                # Total evolution time

# Time grid
tlist = collect(range(0, T, length=n_steps+1))

# ===== DEFINE CONTROL PULSES =====
# Initial guess: smooth pulses using flattop shape

# Control amplitudes (initial guesses)
function ϵ_Va1(t)
    return 0.5 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Va2(t)
    return 0.3 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Va3(t)
    return 0.4 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Vb1(t)
    return 0.5 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Vb2(t)
    return 0.3 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Vb3(t)
    return 0.4 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_U(t)
    return 1.0 * QuantumControl.Shapes.flattop(t, T=T, t_rise=0.5, func=:blackman)
end

function ϵ_Ja(t)
    return 1.0 
end

function ϵ_Jb(t)
    return 1.0 
end

# ===== CONSTRUCT HAMILTONIAN =====
# H = H0 + sum_k ϵ_k(t) * H_k
# Control Hamiltonians: relative site potentials + interaction + hopping

# Drift Hamiltonian (zero)
H0 = zeros(ComplexF64, 16, 16)

# Control Hamiltonians
H_Va1 = P_a_sites[2] - P_a_sites[1]  # Va site 1 relative to site 0
H_Va2 = P_a_sites[3] - P_a_sites[1]  # Va site 2 relative to site 0
H_Va3 = P_a_sites[4] - P_a_sites[1]  # Va site 3 relative to site 0
H_Vb1 = P_b_sites[2] - P_b_sites[1]  # Vb site 1 relative to site 0
H_Vb2 = P_b_sites[3] - P_b_sites[1]  # Vb site 2 relative to site 0
H_Vb3 = P_b_sites[4] - P_b_sites[1]  # Vb site 3 relative to site 0
H_U = Hint_base                       # Interaction
H_Ja = Hhop_a                         # Hopping a
H_Jb = Hhop_b                         # Hopping b

# Build full Hamiltonian with QuantumControl
H = hamiltonian(
    H0,
    (H_Va1, ϵ_Va1),
    (H_Va2, ϵ_Va2),
    (H_Va3, ϵ_Va3),
    (H_Vb1, ϵ_Vb1),
    (H_Vb2, ϵ_Vb2),
    (H_Vb3, ϵ_Vb3),
    (H_U, ϵ_U),
    (H_Ja, ϵ_Ja),
    (H_Jb, ϵ_Jb)
)

# ===== INITIAL AND TARGET STATES =====
# Initial state: |0,1⟩ (particle a at site 0, particle b at site 1)
psi0 = zeros(ComplexF64, 16)
psi0[idx(0, 1)] = 1.0

# Target state: (|0,1⟩ + |2,3⟩)/sqrt(2) - SPDC-like entangled state
psi_target = zeros(ComplexF64, 16)
psi_target[idx(0, 1)] = 1.0 / sqrt(2)
psi_target[idx(2, 3)] = 1.0 / sqrt(2)

println("Initial state: |0,1⟩")
println("Target state: (|0,1⟩ + |2,3⟩)/√2")

# ===== SETUP CONTROL PROBLEM =====
trajectories = [Trajectory(initial_state=psi0, generator=H, target_state=psi_target)]

using QuantumControl.Functionals: J_T_ss

problem = ControlProblem(
    trajectories=trajectories,
    tlist=tlist,
    iter_stop=500,
    prop_method=ExpProp,
    J_T=J_T_ss,
    check_convergence=res -> begin
        ((res.J_T < 1e-5) && (res.converged = true) && (res.message = "J_T < 10⁻⁵"))
    end,
)

println("\nSetting up GRAPE optimization...")
println("  Time steps: $n_steps")
println("  Total time: $T")
println("  Time grid points: $(length(tlist))")

# ===== RUN GRAPE OPTIMIZATION =====
println("\nRunning GRAPE optimization...")

opt_result = optimize(problem; method=GRAPE, prop_method=ExpProp)

println("\n===== OPTIMIZATION COMPLETE =====")
println("Final J_T: $(opt_result.J_T)")
println("Iterations: $(opt_result.iter)")
println("Converged: $(opt_result.converged)")
if !isempty(opt_result.message)
    println("Message: $(opt_result.message)")
end

# ===== EXTRACT OPTIMIZED CONTROLS =====
optimized_controls = opt_result.optimized_controls

# Sample the optimized controls at the time grid
dt = T / n_steps
Va_opt = zeros(n_steps, 4)
Vb_opt = zeros(n_steps, 4)
U_opt = zeros(n_steps)
Ja_opt = zeros(n_steps)
Jb_opt = zeros(n_steps)

for k in 1:n_steps
    # GRAPE uses relative potentials: H_Va1 = P[2] - P[1], etc.
    # This means site 0 accumulates the negative sum of all relative controls
    # to correctly reproduce the GRAPE Hamiltonian in HEngg_ode.jl

    # Va controls (relative to site 0)
    Va1 = optimized_controls[1][k]
    Va2 = optimized_controls[2][k]
    Va3 = optimized_controls[3][k]
    Va_opt[k, 1] = -(Va1 + Va2 + Va3)  # Site 0 gets negative sum
    Va_opt[k, 2] = Va1
    Va_opt[k, 3] = Va2
    Va_opt[k, 4] = Va3

    # Vb controls (relative to site 0)
    Vb1 = optimized_controls[4][k]
    Vb2 = optimized_controls[5][k]
    Vb3 = optimized_controls[6][k]
    Vb_opt[k, 1] = -(Vb1 + Vb2 + Vb3)  # Site 0 gets negative sum
    Vb_opt[k, 2] = Vb1
    Vb_opt[k, 3] = Vb2
    Vb_opt[k, 4] = Vb3

    # Interaction and hopping
    U_opt[k] = optimized_controls[7][k]
    Ja_opt[k] = optimized_controls[8][k]
    Jb_opt[k] = optimized_controls[9][k]
end

# ===== SAVE PULSE =====
output_file = "GRAPE_pulse.jld2"
final_fidelity = 1.0 - opt_result.J_T

@save output_file n=n_steps T=T dt=dt Va=Va_opt Vb=Vb_opt U=U_opt Ja=Ja_opt Jb=Jb_opt fidelity=final_fidelity

println("\nPulse saved to: $output_file")
println("  n = $n_steps time steps")
println("  T = $T total time")
println("  dt = $dt step size")
println("  Final fidelity ≈ $final_fidelity")
