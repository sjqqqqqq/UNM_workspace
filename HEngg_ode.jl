# test_ode.jl
# Compare matrix exponential evolution vs ODE evolution
# Uses optimized pulse from HEngg_qCOMBAT.jl

using LinearAlgebra
using DifferentialEquations
using Plots

# Include the HEngg_qCOMBAT.jl to get the optimized pulse
include("HEngg_qCOMBAT.jl")

println("\n" * "="^60)
println("ODE EVOLUTION TEST")
println("="^60)

# Ensure we have the optimized pulse
if isnothing(pulse)
    error("No optimized pulse found! Run HEngg_qCOMBAT.jl first.")
end

println("\nUsing optimized pulse from HEngg_qCOMBAT.jl")
println("Number of steps: ", pulse.n)
println("Total time T: ", pulse.T)
println("Time step dt: ", pulse.dt)

# Define initial and target states (same as in HEngg_qCOMBAT.jl)
psi0 = zeros(ComplexF64, 16)
psi0[idx(0, 1)] = 1.0

psi_target = zeros(ComplexF64, 16)
psi_target[idx(0, 1)] = 1.0 / sqrt(2)
psi_target[idx(2, 3)] = 1.0 / sqrt(2)

# Function to get the Hamiltonian at a given time
function H_at_time(t::Float64, p::PulseFull)
    # Determine which time step we're in
    m = floor(Int, t / p.dt)

    # Make sure we don't go past the last step
    if m >= 2 * p.n
        m = 2 * p.n - 1
    end

    return H_step(p, m)
end

# ODE function: dψ/dt = -iH(t)ψ
function schrodinger!(dpsi, psi, params, t)
    p = params
    H = H_at_time(t, p)
    dpsi .= -1im * H * psi
end

# Solve using ODE solver
println("\nSolving Schrodinger equation using ODE solver...")
tspan = (0.0, pulse.T)
prob = ODEProblem(schrodinger!, psi0, tspan, pulse)

# Use a high-order solver for accuracy
sol = solve(prob, Tsit5(), saveat=collect(t_grid), abstol=1e-10, reltol=1e-10)

println("ODE solution completed!")

# Extract the final state and all intermediate states
psi_final_ode = sol.u[end]
psis_ode = sol.u

# Compute fidelities for ODE evolution
F_ode = [abs(dot(psi_target, psi))^2 for psi in psis_ode]

println("\nODE Evolution Results:")
println("Final fidelity (ODE): ", F_ode[end])

# Now compute using matrix exponential (from the optimized pulse)
println("\nComputing evolution using matrix exponential...")
opt = GRAPEAdam(pulse)
psis_matexp, Us = forward(opt, psi0)
psi_final_matexp = psis_matexp[end]

# Compute fidelities for matrix exponential
F_matexp = [abs(dot(psi_target, psi))^2 for psi in psis_matexp]

println("Final fidelity (Matrix Exp): ", F_matexp[end])

# Compare the two methods
println("\n" * "="^60)
println("COMPARISON")
println("="^60)

# Fidelity difference
fidelity_diff = abs(F_ode[end] - F_matexp[end])
println("Fidelity difference: ", fidelity_diff)

# State overlap
state_overlap = abs(dot(psi_final_ode, psi_final_matexp))^2
println("Final state overlap: ", state_overlap)

# Max fidelity difference over time
max_fid_diff = maximum(abs.(F_ode .- F_matexp))
println("Max fidelity difference over time: ", max_fid_diff)

# Plot comparison
fig_comparison = plot(layout=(2,1), size=(800, 800))

# Subplot 1: Fidelities
plot!(fig_comparison[1], collect(t_grid), F_matexp, label="Matrix Exp",
      linewidth=2, color=:blue, xlabel="Time", ylabel="Fidelity",
      title="Fidelity Comparison")
plot!(fig_comparison[1], collect(t_grid), F_ode, label="ODE",
      linewidth=2, color=:red, linestyle=:dash)

# Subplot 2: Difference
plot!(fig_comparison[2], collect(t_grid), abs.(F_ode .- F_matexp),
      label="", linewidth=2, color=:black, xlabel="Time",
      ylabel="|F_ODE - F_MatExp|", title="Fidelity Difference",
      yscale=:log10)

savefig(fig_comparison, "comparison_ode_vs_matexp.png")
println("\nComparison plot saved: comparison_ode_vs_matexp.png")

# Print detailed comparison at a few time points
println("\nDetailed comparison at selected times:")
println("Time\t\tF_MatExp\tF_ODE\t\tDifference")
println("-"^60)
selected_indices = [1, length(t_grid)÷4, length(t_grid)÷2, 3*length(t_grid)÷4, length(t_grid)]
for i in selected_indices
    t_val = t_grid[i]
    println("$(round(t_val, digits=4))\t\t$(round(F_matexp[i], digits=10))\t$(round(F_ode[i], digits=10))\t$(round(abs(F_ode[i] - F_matexp[i]), digits=12))")
end

println("\n" * "="^60)
println("TEST COMPLETE")
println("="^60)
