# HEngg_ode.jl
# Time evolution from input pulse - no optimization
# Input a pulse and get the time evolution using ODE solver

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
    bounds::Dict{Symbol, Tuple{Float64, Float64}}

    function PulseFull(n::Int, T::Real; pulse_file::String="optimized_pulse.jld2")
        n = Int(n)
        T = Float64(T)
        dt = T / (2 * n)

        # Load optimized pulse from file
        if isfile(pulse_file)
            data = load(pulse_file)
            Va = data["Va"]
            Vb = data["Vb"]
            U = data["U"]
            Ja = data["Ja"]
            Jb = data["Jb"]
        else
            error("Pulse file not found: $pulse_file. Please run HEngg_qCOMBAT.jl first to generate the optimized pulse.")
        end

        bounds = Dict{Symbol, Tuple{Float64, Float64}}(
            :Va => (-3.0, 3.0),
            :Vb => (-3.0, 3.0),
            :U => (0.0, 6.0),
            :Ja => (0.0, 5.0),
            :Jb => (0.0, 5.0)
        )

        obj = new(n, T, dt, Va, Vb, U, Ja, Jb, bounds)
        clip!(obj)
        return obj
    end
end

function clip!(p::PulseFull)
    p.Va .= clamp.(p.Va, p.bounds[:Va]...)
    p.Vb .= clamp.(p.Vb, p.bounds[:Vb]...)
    p.U .= clamp.(p.U, p.bounds[:U]...)
    p.Ja .= clamp.(p.Ja, p.bounds[:Ja]...)
    p.Jb .= clamp.(p.Jb, p.bounds[:Jb]...)
end

function H_step(p::PulseFull, m::Int)
    if m % 2 == 0
        k = m ÷ 2 + 1
        H = zeros(ComplexF64, 16, 16)
        for s in 1:4
            H .+= p.Va[k, s] .* P_a_sites[s]
        end
        for t in 1:4
            H .+= p.Vb[k, t] .* P_b_sites[t]
        end
        H .+= p.U[k] .* Hint_base
        return H
    else
        k = m ÷ 2 + 1
        return p.Ja[k] .* Hhop_a .+ p.Jb[k] .* Hhop_b
    end
end

# ===== TIME EVOLUTION FUNCTIONS =====

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

"""
    evolve_ode(pulse::PulseFull, psi0::Vector{ComplexF64}; n_points=100, abstol=1e-10, reltol=1e-10)

Evolve the initial state psi0 using the given pulse and ODE solver.

# Arguments
- `pulse::PulseFull`: The pulse parameters
- `psi0::Vector{ComplexF64}`: Initial quantum state (normalized)
- `n_points::Int`: Number of time points to save (default: 100)
- `abstol::Float64`: Absolute tolerance for ODE solver (default: 1e-10)
- `reltol::Float64`: Relative tolerance for ODE solver (default: 1e-10)

# Returns
- `times::Vector{Float64}`: Time points
- `states::Vector{Vector{ComplexF64}}`: Quantum states at each time point
"""
function evolve_ode(pulse::PulseFull, psi0::Vector{ComplexF64};
                    n_points::Int=100, abstol::Float64=1e-10, reltol::Float64=1e-10)
    tspan = (0.0, pulse.T)
    saveat = range(0.0, pulse.T, length=n_points)

    prob = ODEProblem(schrodinger!, psi0, tspan, pulse)
    sol = solve(prob, Tsit5(), saveat=saveat, abstol=abstol, reltol=reltol)

    return collect(sol.t), sol.u
end

# ===== EXAMPLE USAGE =====
if abspath(PROGRAM_FILE) == @__FILE__
    # Initial state: |0,1⟩
    psi0 = zeros(ComplexF64, 16)
    psi0[idx(0, 1)] = 1.0

    # Create pulse and evolve
    pulse = PulseFull(36, 2π)
    times, states = evolve_ode(pulse, psi0, n_points=200)

    # Calculate populations and plot
    pops = hcat([abs2.(psi) for psi in states]...)'
    interesting_states = [idx(0,1), idx(1,0), idx(2,3), idx(3,2)]
    labels = ["|0,1⟩", "|1,0⟩", "|2,3⟩", "|3,2⟩"]

    plot(times, pops[:, interesting_states], label=permutedims(labels),
         linewidth=2, xlabel="Time", ylabel="Population",
         title="Population Dynamics", size=(1000, 600))
    savefig("population_dynamics.png")
    println("Plot saved: population_dynamics.png")
end
