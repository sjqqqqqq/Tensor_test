using ITensors, ITensorMPS

let
    # System parameters
    N_sites = 2
    N_particles = 100
    t = 1.0                        # Hopping
    U = 0.1                        # On-site interaction

    # Time evolution parameters
    tau = 0.01
    ttotal = 2.0
    cutoff = 1E-10

    # Create site indices
    s = siteinds("Boson", N_sites; dim=N_particles + 1, conserve_qns=true)

    # Build Trotter gates
    gates = ITensor[]

    # Forward hopping gates
    for j in 1:(N_sites - 1)
        hj = -t * (op("Adag", s[j]) * op("A", s[j+1]) +
                   op("A", s[j]) * op("Adag", s[j+1]))
        push!(gates, exp(-im * tau / 2 * hj))
    end

    # On-site interaction gates
    for j in 1:N_sites
        hj = (U/2) * (op("N * N", s[j]) - op("N", s[j]))
        push!(gates, exp(-im * tau * hj))
    end

    # Reverse hopping gates
    for j in (N_sites-1):-1:1
        hj = -t * (op("Adag", s[j]) * op("A", s[j+1]) +
                   op("A", s[j]) * op("Adag", s[j+1]))
        push!(gates, exp(-im * tau / 2 * hj))
    end

    # Initial state: all particles on site 1
    psi = MPS(s, ["$N_particles", "0"])

    println("Time\tSite1\tSite2\tTotal")

    # Time evolution
    for t in 0.0:tau:ttotal
        n1, n2 = expect(psi, "N"; sites=1), expect(psi, "N"; sites=2)
        println("$t\t$(round(n1, digits=2))\t$(round(n2, digits=2))\t$(round(n1+n2, digits=2))")

        t ≈ ttotal && break

        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end
end
