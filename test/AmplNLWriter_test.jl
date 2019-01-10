using JuMP, AmplNLWriter, MathOptInterfaceToMathProgBase, Test, Ipopt
using Random, Statistics, Test

const MOI = JuMP.MOI

model = JuMP.Model(
    with_optimizer(
        MathOptInterfaceToMathProgBase.Optimizer,
        solver = AmplNLWriter.AmplNLSolver(Ipopt.amplexe)
    )
)

N = 1000
h = 1/N
alpha = 350

@variables(model, begin
    -1 <= t[1:(N + 1)] <= 1
    -0.05 <= x[1:(N + 1)] <= 0.05
    u[1:(N + 1)]
end)
@NLobjective(model, Min, sum(0.5 * h * (u[i + 1]^2 + u[i]^2) +
    0.5 * alpha * h * (cos(t[i + 1]) + cos(t[i])) for i in 1:N)
)
@NLconstraint(model, [i = 1:N],
    x[i + 1] - x[i] - 0.5 * h * (sin(t[i + 1]) + sin(t[i])) == 0
)
@constraint(model, [i = 1:N],
    t[i + 1] - t[i] - 0.5 * h * u[i + 1] - 0.5 * h * u[i] == 0
)
JuMP.optimize!(model)

@test JuMP.termination_status(model) == MOI.OPTIMAL
@test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
@test JuMP.objective_value(model) ≈ 350.0

function example_mle(; verbose = true)
    n = 1_000
    Random.seed!(1234)
    data = randn(n)

    model = JuMP.Model(
        with_optimizer(
            MathOptInterfaceToMathProgBase.Optimizer,
            solver = AmplNLWriter.AmplNLSolver(Ipopt.amplexe)
        )
    )

    @variable(model, μ, start = 0.0)
    @variable(model, σ >= 0.0, start = 1.0)
    @NLobjective(model, Max, n / 2 * log(1 / (2 * π * σ^2)) -
        sum((data[i] - μ)^2 for i in 1:n) / (2 * σ^2)
    )
    JuMP.optimize!(model)
    if verbose
        println("μ = ", JuMP.value(μ))
        println("mean(data) = ", mean(data))
        println("σ^2 = ", JuMP.value(σ)^2)
        println("var(data) = ", var(data))
        println("MLE objective: ", JuMP.objective_value(m))
    end
    @test JuMP.value(μ) ≈ mean(data) atol = 1e-3
    @test JuMP.value(σ)^2 ≈ var(data) atol = 1e-2

    # constrained MLE?
    @NLconstraint(model, μ == σ^2)

    JuMP.optimize!(model)
    if verbose
        println("\nWith constraint μ == σ^2:")
        println("μ = ", JuMP.value(μ))
        println("σ^2 = ", JuMP.value(σ)^2)
        println("Constrained MLE objective: ", JuMP.objective_value(model))
    end
    @test JuMP.value(μ) ≈ JuMP.value(σ)^2
end

example_mle(verbose = false)
