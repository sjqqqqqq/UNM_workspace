# test_ode.jl
# Compare matrix exponential evolution vs ODE evolution
# Self-contained version with GRAPE optimization included

using LinearAlgebra
using DifferentialEquations
using Plots
using Random

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

    function PulseFull(n::Int, T::Real; seed::Int=11)
        n = Int(n)
        T = Float64(T)
        dt = T / (2 * n)

        Va = [-0.7428595944616008 -0.00144427511977008 0.20299671524671492 -0.9426219832561109;
              -0.7041478308450881 0.856422045920739 -0.8591588476916063 -0.740452101201404;
              0.8966569065835501 0.24376718559276567 -0.26201375254041803 0.0227800436065253;
              0.3256859050335985 -0.4493823684777414 -0.7240638542660893 0.5760791890079837;
              0.34072116820496756 0.02476462696632087 0.6334728719393161 0.09815053774005267;
              0.9618272785946109 -0.5909810773399102 0.10746072573045562 -0.03275060615322456;
              -0.2934502898791038 0.18319060789808694 -0.5293975366648389 0.6044053675569669;
              0.7346671036848293 -0.7424806575442979 -0.06585358652345774 -0.44571021493259133;
              -0.8337660045295152 0.791888616500735 -0.14010261590432926 -0.7046174000758119;
              0.3467247145518215 -0.5955679444239237 0.8028621573763539 -0.5657034851629437;
              -0.9338506252451426 -0.5984620916775614 -0.30850425248872093 -0.06218367315503071;
              0.8122686777911212 0.3947221774725955 -0.32135867851510946 -0.9662455698050043;
              -0.680352611790038 0.992871751079341 -0.08056804021363084 0.38207983263751855;
              -0.890663877196324 -0.9318994421059248 0.6917802128901149 0.175763881333721;
              -0.38258051359087597 -0.3652467234373251 -0.8215254911645025 -0.6546607977828491;
              -0.9508277850696274 0.6782496967455633 -0.06739360559366969 -0.7455941678829392;
              0.4784937480673841 -0.6086943401093581 -0.8761595297030949 0.19678421464807627;
              0.7915155034825632 -0.9461131772305944 0.6102719797833385 -0.6196599665433185;
              -0.8141971567269057 -0.9640759286948877 -0.41404986597582805 0.45422348796383627;
              -0.01364209798376614 0.7058399889090554 -0.5655773929570194 -0.3696345057131367;
              -0.4837182962908233 0.956602274628434 0.8820119893028697 -0.31862764828698187;
              -0.12799699281533794 -0.37136074259229424 0.49301783461082116 -0.9199737948936195;
              -0.8651221338159478 -0.19192482506318753 -0.509811897771872 0.6904499634024537;
              0.48361419295058017 0.09158796504591349 0.3229510286270012 0.38455821336984775;
              0.5621096467966842 0.8550051942313655 -0.7005296524245341 0.252260315490763;
              -0.7127571137732898 -0.11373811743060624 0.5725695772474639 0.7893955873722909;
              0.5184563082490594 -0.9291835918368125 -0.2811758471324466 -0.6738863351569422;
              0.9976049752648035 -0.7119692847399253 -0.5113711143070341 -0.28556057444094596;
              -0.8782265941816241 0.7407698340866675 0.27272284372624855 -0.6805035200831602;
              -0.00346288457489963 -0.8426604827300848 0.22195957471979133 -0.5366735438050847;
              -0.922722134270483 -0.769524992612195 0.11052524524044216 0.2739612219163643;
              -0.3503987057316058 0.28690485912217367 -0.2958678001827746 -0.7386819864195344;
              -0.3695794361532656 -0.20946354749236096 0.8252776694061896 -0.7686792525604873;
              -0.827672218865493 0.12300405414342563 0.9267855769396518 0.8145102525135857;
              0.40048811556914976 -0.8664985067086894 0.6128028485617785 0.3667494747643152;
              -0.7121395214981432 -0.07011254794525734 -0.9013546892409319 0.6037519305692991]

        Vb = [0.4372684343765658 0.6092561998361057 0.5208312812403544 -0.4654848460746053;
              0.5794810289870223 -0.5016587081591872 -0.724209044127214 -0.21921993278170437;
              -0.00431929421794375 -0.42835570472605866 0.21156580910783518 0.20500745841878887;
              -0.5203646835713671 0.24513497566529763 -0.28549416489164714 0.46937919730750055;
              -0.4191640967621526 0.5975868576807413 -0.16977903822918106 0.1064804688527452;
              0.34669264337607 0.03675475100289249 -0.48476366912392255 0.9587631729908321;
              -0.8078891862253117 -0.3501526477668935 0.09822459616487489 -0.94259905547406;
              -0.6901738278159486 0.5985288371211959 0.5676287639901312 0.2739327942462133;
              0.8035993201887757 0.5110468066741114 -0.4024310661021575 0.28973305242528546;
              -0.31983527128158906 0.4996888658483529 -0.23041258231106854 -0.6935132824838135;
              0.752913216451421 0.38081089460720174 0.48935747830862475 0.11919582900698633;
              0.5658832355939718 -0.1042532072426059 0.13158608132648641 -0.874872435214997;
              0.1101374282768568 0.6292071998356119 0.411091047437939 0.6063043906584094;
              -0.00780176068767058 0.7626702722254766 -0.7833953236581142 0.7514010595643232;
              -0.25745701052428704 -0.8180749365681097 0.23866603698040234 -0.09021478984116493;
              -0.14325479493453375 0.7040580824291109 -0.7244172651199623 0.23386961862593636;
              -0.17249970315707208 0.05651750987381243 -0.00118123873571241 -0.7317418497441599;
              0.02403076405215576 0.7242109809482165 -0.657531509485255 -0.977062216540949;
              -0.8648694064734579 -0.08040859372979825 0.9489041073831945 -0.9115262020249544;
              0.98185902681229 0.07210616448682639 -0.7597194561131462 -0.16292124996453783;
              -0.5852798591114456 0.42830293618958626 0.08302738093118411 -0.4240800956579409;
              -0.48915472659311776 0.7339537460733248 0.532484048089864 -0.12762991585031846;
              -0.18897753319436328 0.47494581273127534 0.9414692579115818 -0.8408147367465124;
              -0.6818327671364399 -0.27534660035255354 0.0154295874836794 -0.8571825061196054;
              -0.8930207752856092 -0.5610754968380365 -0.22808095168576448 0.47856384225672954;
              0.2198286287921507 -0.9415918091001985 -0.9099742930077774 -0.09594054770313987;
              0.7497013066768781 0.8299942352151819 -0.2697891433807953 0.7747453183559929;
              0.8582486167292582 -0.18677197243209287 0.49214632340204867 -0.2693153071778429;
              -0.6928459755105352 0.15039591040158817 -0.8271553754473717 0.3267063158833985;
              0.6183998124861247 0.8308118859215743 -0.10395002996046565 -0.7652304015216784;
              0.8049830278448147 0.7407398406142378 0.9352204360526384 0.18795769860552625;
              0.3468071082802173 -0.253441044104437 -0.6337280448871927 -0.41656997448686983;
              0.4413130786682353 -0.3500327313377316 0.38317295923777217 0.00431841831818769;
              -0.10200781650865576 0.935255804893369 -0.6678189668063357 -0.0256784124074676;
              -0.6805707187127903 0.8749139426375705 -0.02276117601134597 -0.33606796872034694;
              -0.6633594135990877 0.18820139861706942 -0.20208214895857712 -0.7456397291004577]

        U = [0.2582727539179657, 0.9043446798483925, 1.2497031676590646,
             1.13388519187877, 1.820049945330154, 1.8409133044389672,
             1.8853353849640726, 1.6394467693053487, 1.0597084042930598,
             1.142831212535777, 0.9151671078672106, 0.9191788262794909,
             0.2748434138540008, 1.6378804477203912, 0.6428365534973408,
             0.2528102790087004, 1.0396628351425696, 1.492141160434744,
             0.9848572221900056, 0.855439917036747, 1.3853487606130364,
             0.49313855973241116, 0.22039738133256806, 1.2663614266483023,
             1.1548109649882765, 1.7584769086506815, 0.9485663623637273,
             1.6150075393075118, 0.23226653324622853, 0.249100130135643,
             1.277417805197495, 0.60723138516855, 0.3210778583139092,
             0.429949221542521, 0.8745375315661452, 0.8068464775245257]

        Ja = [0.5637924893785216, 0.8875952983608646, 0.36567756934138707,
              0.1841566613119261, 0.5780302477783276, 0.00215777817948348,
              0.38481836635411126, 0.317236887598234, 0.38739126684177394,
              0.05080103096596422, 0.554216886117733, 0.6241677145870826,
              0.7494654678181312, 0.9574536785617811, 0.35474262218035635,
              0.6562901765015364, 0.442042218364359, 0.2150477140696757,
              0.8892952152839733, 0.7459063823156618, 0.9427888976125194,
              0.8215420262975919, 0.9174680145841102, 0.12804791278567007,
              0.01571453572525505, 0.19858722946932605, 0.41212509429159216,
              0.6876430157240668, 0.4384611833527342, 0.513346280126808,
              0.7609921126035006, 0.07060343625365229, 0.6783002180032774,
              0.04872865838077467, 0.24864853158753608, 0.7222737735973856]

        Jb = [0.6830682599050896, 0.4052864896569335, 0.29455767458146465,
              0.8281077620045911, 0.2747533889489221, 0.03121134819157656,
              0.24726729061856922, 0.02119338174046981, 0.8106908479800631,
              0.12226983764226085, 0.9739700411195548, 0.21298368639510035,
              0.3060714149976571, 0.5851922862553449, 0.30820810705082513,
              0.3792938853014244, 0.10421019388155572, 0.3890709874035916,
              0.02523293988788589, 0.22396692242099092, 0.9094703295567117,
              0.2807764740442198, 0.27089834498352905, 0.6792624397768348,
              0.8829032712488051, 0.2958443441484473, 0.37632881322022116,
              0.7146723547095178, 0.7921459755036756, 0.6137939122560816,
              0.17899740667351782, 0.18073532023835748, 0.7579526866549317,
              0.706829739510714, 0.896278924358489, 0.7798000832637796]

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

function unitaries(p::PulseFull)
    return [exp(-1im * H_step(p, m) * p.dt) for m in 0:(2*p.n-1)]
end

# ===== GRAPE OPTIMIZER =====
mutable struct GRAPEAdam
    p::PulseFull
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    l2::Float64
    m::Dict{Symbol, Array{Float64}}
    v::Dict{Symbol, Array{Float64}}
    t::Int

    function GRAPEAdam(pulse::PulseFull; lr=0.08, beta1=0.9, beta2=0.999, eps=1e-8, l2=1e-5)
        m = Dict{Symbol, Array{Float64}}(
            :Va => zeros(size(pulse.Va)),
            :Vb => zeros(size(pulse.Vb)),
            :U => zeros(size(pulse.U)),
            :Ja => zeros(size(pulse.Ja)),
            :Jb => zeros(size(pulse.Jb))
        )
        v = Dict{Symbol, Array{Float64}}(
            k => zeros(size(val)) for (k, val) in m
        )
        new(pulse, lr, beta1, beta2, eps, l2, m, v, 0)
    end
end

function forward(opt::GRAPEAdam, psi0::Vector{ComplexF64})
    Us = unitaries(opt.p)
    psis = [psi0]
    for U in Us
        push!(psis, U * psis[end])
    end
    return psis, Us
end

function gradients(opt::GRAPEAdam, psi0::Vector{ComplexF64}, psi_target::Vector{ComplexF64})
    n = opt.p.n
    dt = opt.p.dt

    psis, Us = forward(opt, psi0)
    psiT = psis[end]
    overlap = dot(psi_target, psiT)
    F = real(overlap * conj(overlap))

    chis = Vector{Union{Nothing, Vector{ComplexF64}}}(nothing, 2*n+1)
    chis[end] = psi_target .* conj(overlap)

    for m in (2*n-1):-1:0
        chis[m+1] = transpose(conj(Us[m+1])) * chis[m+2]
    end

    gVa = zeros(size(opt.p.Va))
    gVb = zeros(size(opt.p.Vb))
    gU = zeros(size(opt.p.U))
    gJa = zeros(size(opt.p.Ja))
    gJb = zeros(size(opt.p.Jb))

    for m in 0:(2*n-1)
        k = m ÷ 2 + 1
        psi_m1 = psis[m+2]
        chi_mp1 = chis[m+2]
        fac = conj(overlap)

        if m % 2 == 0
            for s in 1:4
                cont = -1im * dt * dot(chi_mp1, P_a_sites[s] * psi_m1)
                gVa[k, s] += 2.0 * real(cont * fac)
            end
            for t in 1:4
                cont = -1im * dt * dot(chi_mp1, P_b_sites[t] * psi_m1)
                gVb[k, t] += 2.0 * real(cont * fac)
            end
            cont = -1im * dt * dot(chi_mp1, Hint_base * psi_m1)
            gU[k] += 2.0 * real(cont * fac)
        else
            cont_a = -1im * dt * dot(chi_mp1, Hhop_a * psi_m1)
            gJa[k] += 2.0 * real(cont_a * fac)
            cont_b = -1im * dt * dot(chi_mp1, Hhop_b * psi_m1)
            gJb[k] += 2.0 * real(cont_b * fac)
        end
    end

    if opt.l2 > 0
        gVa .-= 2 * opt.l2 .* opt.p.Va
        gVb .-= 2 * opt.l2 .* opt.p.Vb
        gU .-= 2 * opt.l2 .* opt.p.U
        gJa .-= 2 * opt.l2 .* opt.p.Ja
        gJb .-= 2 * opt.l2 .* opt.p.Jb
    end

    grads = Dict{Symbol, Array{Float64}}(
        :Va => gVa, :Vb => gVb, :U => gU, :Ja => gJa, :Jb => gJb
    )

    return F, overlap, grads
end

function adam_step!(opt::GRAPEAdam, key::Symbol, g::Array{Float64})
    opt.t += 1

    m = opt.m[key]
    v = opt.v[key]

    m .= opt.beta1 .* m .+ (1 - opt.beta1) .* g
    v .= opt.beta2 .* v .+ (1 - opt.beta2) .* (g .^ 2)

    m_hat = m ./ (1 - opt.beta1^opt.t)
    v_hat = v ./ (1 - opt.beta2^opt.t)

    return opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.eps)
end

function optimize!(opt::GRAPEAdam, psi0::Vector{ComplexF64}, psi_target::Vector{ComplexF64}; iters=360)
    hist = Float64[]

    for it in 1:iters
        F, ov, grads = gradients(opt, psi0, psi_target)
        push!(hist, F)

        if it % 50 == 0 || it <= 5 || it > 355
            println("  Iteration $(it-1): Fidelity = $(round(F, digits=10))")
        end

        for key in [:Va, :Vb, :U, :Ja, :Jb]
            step = adam_step!(opt, key, grads[key])
            if key == :Va
                opt.p.Va .+= step
            elseif key == :Vb
                opt.p.Vb .+= step
            elseif key == :U
                opt.p.U .+= step
            elseif key == :Ja
                opt.p.Ja .+= step
            elseif key == :Jb
                opt.p.Jb .+= step
            end
        end

        clip!(opt.p)
    end

    return hist
end

function fidelity_vs_time(opt::GRAPEAdam, psi0::Vector{ComplexF64}, psi_target::Vector{ComplexF64})
    psis, _ = forward(opt, psi0)
    times = range(0.0, opt.p.T, length=2*opt.p.n+1)
    F_t = [abs(dot(psi_target, psi))^2 for psi in psis]
    return times, F_t
end

# ===== OPTIMIZATION SETUP =====
println("="^60)
println("RUNNING GRAPE OPTIMIZATION")
println("="^60)

psi0 = zeros(ComplexF64, 16)
psi0[idx(0, 1)] = 1.0

psi_target = zeros(ComplexF64, 16)
psi_target[idx(0, 1)] = 1.0 / sqrt(2)
psi_target[idx(2, 3)] = 1.0 / sqrt(2)

T = 2 * π
n = 36
pulse = PulseFull(n, T, seed=11)
opt = GRAPEAdam(pulse, lr=0.08, l2=1e-5)

println("\nRunning optimization (360 iterations)...")
optimize!(opt, psi0, psi_target, iters=360)

t_grid, F_t = fidelity_vs_time(opt, psi0, psi_target)

println("\nOptimization complete!")
println("Final fidelity: ", F_t[end])

# ===== ODE EVOLUTION TEST =====
println("\n" * "="^60)
println("ODE EVOLUTION TEST")
println("="^60)

println("\nUsing optimized pulse")
println("Number of steps: ", pulse.n)
println("Total time T: ", pulse.T)
println("Time step dt: ", pulse.dt)

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
