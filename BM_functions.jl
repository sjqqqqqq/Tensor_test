"""
Generate Fock basis for N bosons in k sites.
Returns basis vectors and index dictionary.
"""
function make_basis(N::Int, k::Int)
    # Generate all ways to place N bosons in k sites
    basis = Vector{Vector{Int}}()

    # Using stars and bars: choose k-1 positions from N+k-1
    for combo in combinations(0:(N+k-2), k-1)
        state = zeros(Int, k)
        prev = -1
        for (i, pos) in enumerate(combo)
            state[i] = pos - prev - 1
            prev = pos
        end
        state[k] = N + k - 2 - prev
        push!(basis, state)
    end

    # Sort to match Python ordering (reversed)
    reverse!(basis)

    # Create index dictionary
    ind = Dict{Vector{Int}, Int}()
    for (i, state) in enumerate(basis)
        ind[state] = i
    end

    return basis, ind
end

"""
Compute Hilbert space dimension for N bosons in k sites.
"""
hilbert_dim(N::Int, k::Int) = binomial(N + k - 1, k - 1)

# ============================================================================
# Hamiltonian matrix construction
# ============================================================================

"""
Construct Hopping matrix for Bose-Hubbard model.
H_hop = sum_j (a†_j a_{j+1} + h.c.)
"""
function hopping_matrix(basis::Vector{Vector{Int}}, ind::Dict{Vector{Int}, Int})
    d = length(basis)
    k = length(basis[1])
    H = zeros(ComplexF64, d, d)

    for (i, state) in enumerate(basis)
        for site in 1:(k-1)
            if state[site] > 0
                new_state = copy(state)
                new_state[site] -= 1
                new_state[site+1] += 1

                if haskey(ind, new_state)
                    j = ind[new_state]
                    val = sqrt(state[site] * new_state[site+1])
                    H[i, j] = val
                    H[j, i] = val
                end
            end
        end
    end

    return H
end

"""
Construct Sz (tilt/detuning) matrix.
Sz = sum_j (j - (k-1)/2) * n_j
"""
function sz_matrix(basis::Vector{Vector{Int}})
    d = length(basis)
    k = length(basis[1])
    sz_diag = zeros(Float64, d)

    for (i, state) in enumerate(basis)
        for (j, n) in enumerate(state)
            sz_diag[i] += n * (j - 1 - (k-1)/2)
        end
    end

    return Diagonal(sz_diag)
end

"""
Construct Interaction matrices for each site.
Returns vector of diagonal matrices for n_j(n_j - 1).
"""
function interaction_matrices(basis::Vector{Vector{Int}})
    d = length(basis)
    k = length(basis[1])

    # Single interaction matrix: sum over all sites of n_j(n_j - 1)
    int_diag = zeros(Float64, d)

    for (i, state) in enumerate(basis)
        for n in state
            int_diag[i] += n * (n - 1)
        end
    end

    return Diagonal(int_diag)
end
