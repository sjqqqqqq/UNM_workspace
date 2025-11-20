using OptimalControl
using NLPModelsIpopt
using Plots

ocp = @def begin
    t ∈ [0, 10], time
    x ∈ R², state
    u ∈ R, control
    g = 9.81
    L = 2.0
    x(0) == [2π/3, 0]
    ẋ(t) == [x₂(t), g*sin(x₁(t))/L - u(t)*cos(x₁(t))/L]
    ( x₁(10)^2 + x₂(10)^2 ) + ∫( 0.5u(t)^2 ) → min
end

sol = solve(ocp)
plot(sol)