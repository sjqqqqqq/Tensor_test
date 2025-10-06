using QuantumOptics
using PyPlot

# Parameters
t = [0:0.1:5;]

# Hamiltonian
b = SpinBasis(1//2)
H = sigmaz(b) ⊗ sigmaz(b) + 1/2 * sigmap(b) ⊗ sigmam(b) + 1/2 * sigmam(b) ⊗ sigmap(b) 

ψ₀ = spinup(b) ⊗ spindown(b)
tout, ψₜ = timeevolution.schroedinger(t, ψ₀, H)

Sz = expect(1, sigmaz(b), ψₜ)
figure(figsize=(6, 3))
plot(tout, Sz)