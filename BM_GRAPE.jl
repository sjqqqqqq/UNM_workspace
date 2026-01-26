# BM_GRAPE.jl
# GRAPE optimal control for Bose-Hubbard model
# Transfer from |N,0,...,0> to (|N,0,...,0> + |0,...,0,N>)/sqrt(2)

using LinearAlgebra
using Combinatorics
using Printf
using Random

include("BM_fuctions.jl")


# ============================================================================
# GRAPE implementation using GRAPE.jl package
# ============================================================================

using QuantumControl
using QuantumControl: Trajectory
using QuantumControl.Functionals: J_T_ss
using QuantumPropagators: hamiltonian, ExpProp, propagate
using QuantumPropagators.Controls: get_controls
using GRAPE

"""
Construct time-dependent Hamiltonian for GRAPE.jl.
H(t) = J(t)*H_hop + U(t)*H_int + Delta(t)*H_sz

Returns a Hamiltonian object compatible with QuantumPropagators.
"""
function make_grape_hamiltonian(H_hop, H_int, H_sz, J_ctrl, U_ctrl, Delta_ctrl)
    # Convert to dense matrices for QuantumPropagators compatibility
    H_hop_dense = Matrix(H_hop)
    H_int_dense = Matrix(H_int)
    H_sz_dense = Matrix(H_sz)

    # Build Hamiltonian: H = J*H_hop + U*H_int + Delta*H_sz
    return hamiltonian(
        (H_hop_dense, J_ctrl),
        (H_int_dense, U_ctrl),
        (H_sz_dense, Delta_ctrl)
    )
end

"""
GRAPE optimization for state transfer using GRAPE.jl package.

Parameters:
- H_hop, H_int, H_sz: Hamiltonian components
- psi0: Initial state
- psi_target: Target state
- T: Total evolution time
- num_steps: Number of time steps
- J0, U0, Delta0: Initial control pulses (vectors of length num_steps)
- max_iter: Maximum number of iterations
- tol: Convergence tolerance (infidelity threshold)

Returns optimized control pulses and optimization result.
"""
function grape_optimize(H_hop, H_int, H_sz, psi0, psi_target, T, num_steps;
                        J0=nothing, U0=nothing, Delta0=nothing,
                        max_iter=1000, tol=1e-6,
                        verbose=true)

    # Time grid
    tlist = collect(range(0, T, length=num_steps))

    # Initialize control pulses (as mutable vectors for GRAPE.jl)
    J_ctrl = isnothing(J0) ? 0.1 * ones(num_steps) : copy(J0)
    U_ctrl = isnothing(U0) ? 0.01 * ones(num_steps) : copy(U0)
    Delta_ctrl = isnothing(Delta0) ? zeros(num_steps) : copy(Delta0)

    # Construct GRAPE-compatible Hamiltonian
    H = make_grape_hamiltonian(H_hop, H_int, H_sz, J_ctrl, U_ctrl, Delta_ctrl)

    # Define trajectory: initial state → target state
    trajectory = Trajectory(
        Vector{ComplexF64}(psi0),
        H;
        target_state=Vector{ComplexF64}(psi_target)
    )

    # Callback for printing progress
    print_callback = verbose ? GRAPE.make_grape_print_iters() : (args...) -> nothing

    # Convergence check: stop when infidelity < tol
    check_conv = res -> (res.J_T < tol) ? "Converged: infidelity = $(res.J_T)" : nothing

    # Run GRAPE optimization
    result = GRAPE.optimize(
        [trajectory],
        tlist;
        prop_method=ExpProp,
        J_T=J_T_ss,
        callback=print_callback,
        iter_stop=max_iter,
        check_convergence=check_conv,
    )

    # Extract optimized controls
    J_opt = result.optimized_controls[1]
    U_opt = result.optimized_controls[2]
    Delta_opt = result.optimized_controls[3]

    # Compute final fidelity
    final_fidelity = 1.0 - result.J_T

    return J_opt, U_opt, Delta_opt, final_fidelity, result
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    # System parameters
    N = 3  # Number of bosons
    k = 3   # Number of sites

    println("="^60)
    println("GRAPE Optimal Control for Bose-Hubbard Model")
    println("="^60)
    println("N = $N bosons, k = $k sites")

    # Build basis
    basis, ind = make_basis(N, k)
    d = length(basis)
    println("Hilbert space dimension: $d")
    println("First basis state: $(basis[1])")
    println("Last basis state: $(basis[end])")

    # Construct Hamiltonian matrices
    println("\nConstructing Hamiltonian matrices...")
    H_hop = hopping_matrix(basis, ind)
    H_sz = sz_matrix(basis)
    H_int = interaction_matrices(basis)

    # Initial state: |N, 0, 0> (all bosons in first site)
    psi0 = zeros(ComplexF64, d)
    psi0[1] = 1.0
    println("Initial state: |$(basis[1])>")

    # Target state: (|N,0,0> + |0,0,N>)/sqrt(2)  (NOON state)
    psi_target = zeros(ComplexF64, d)
    psi_target[1] = 1/sqrt(2)
    psi_target[end] = 1/sqrt(2)
    println("Target state: (|$(basis[1])> + |$(basis[end])>)/√2")

    # Time parameters - longer time for NOON state transfer
    T = 10.0
    num_steps = 1001
    dt = T / (num_steps - 1)
    tspan = range(0, T, length=num_steps)
    println("\nTime: T = $T, steps = $num_steps, dt = $dt")

    # Initial guess for controls with random perturbation to break symmetry
    println("\nStarting GRAPE optimization...")
    println("-"^60)

    Random.seed!(42)  # For reproducibility
    # Use sinusoidal base with random perturbation
    t_arr = collect(tspan)
    J0 = 1.0 .+ 0.5 * sin.(2π * t_arr / T) .+ 0.2 * randn(num_steps)
    U0 = 0.1 .+ 0.05 * cos.(2π * t_arr / T) .+ 0.02 * randn(num_steps)
    Delta0 = 0.5 * sin.(4π * t_arr / T) .+ 0.1 * randn(num_steps)

    # Run GRAPE optimization using GRAPE.jl
    # tol=0.05 corresponds to 95% fidelity (adjust as needed)
    J_opt, U_opt, Delta_opt, final_fidelity, result = grape_optimize(
        H_hop, H_int, H_sz, psi0, psi_target, T, num_steps;
        J0=J0, U0=U0, Delta0=Delta0,
        max_iter=500, tol=1e-4,
        verbose=true
    )

    println("-"^60)
    @printf("Final fidelity: %.8f\n", final_fidelity)
    println("Optimization message: $(result.message)")

    # Save results
    println("\nSaving results...")
    open("J_opt.txt", "w") do f
        for val in J_opt
            println(f, val)
        end
    end
    open("U_opt.txt", "w") do f
        for val in U_opt
            println(f, val)
        end
    end
    open("Delta_opt.txt", "w") do f
        for val in Delta_opt
            println(f, val)
        end
    end
    println("Control pulses saved to J_opt.txt, U_opt.txt, Delta_opt.txt")

    # Verify final state by propagating with optimized controls
    H_opt = make_grape_hamiltonian(H_hop, H_int, H_sz, J_opt, U_opt, Delta_opt)
    tlist = collect(range(0, T, length=num_steps))
    psi_final = propagate(psi0, H_opt, tlist; method=ExpProp)

    println("\nFinal state analysis:")
    println("  |<ψ_target|ψ_final>|² = $(abs2(dot(psi_target, psi_final)))")
    println("  |ψ_final[1]|² = $(abs2(psi_final[1]))")
    println("  |ψ_final[end]|² = $(abs2(psi_final[end]))")

    return J_opt, U_opt, Delta_opt, final_fidelity
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
