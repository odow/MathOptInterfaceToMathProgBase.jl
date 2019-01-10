module MathOptInterfaceToMathProgBase

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathProgBase
const MPB = MathProgBase.SolverInterface

MOIU.@model(InnerModel,
    (MOI.ZeroOne, MOI.Integer),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (),
    (),
    (MOI.SingleVariable,),
    (),
    (),
    ()
)

# Add UniversalFallback to allow the NLPBlock to be set.
const Model = MOIU.UniversalFallback{InnerModel{Float64}}
Base.show(io::IO, ::Model) = println(io, "A MathProgBase model")

"Attribute for the MathProgBase solver."
struct MPBSolver <: MOI.AbstractModelAttribute end

"""
    Optimizer(; solver)

# Example

    model = Optimizer(solver = AmplNLWriter("/usr/bin/bonmin"))
"""
function Optimizer(; solver)
    model = Model()
    MOI.set(model, MPBSolver(), solver)
    return model
end

set_to_bounds(set::MOI.LessThan) = (-Inf, set.upper)
set_to_bounds(set::MOI.GreaterThan) = (set.lower, Inf)
set_to_bounds(set::MOI.EqualTo) = (set.value, set.value)
set_to_bounds(set::MOI.Interval) = (set.lower, set.upper)
set_to_cat(set::MOI.ZeroOne) = :Bin
set_to_cat(set::MOI.Integer) = :Int

function MOI.optimize!(model::Model)
    mpb_solver = MOI.get(model, MPBSolver())

    # Get the variable data.
    variables = MOI.get(model, MOI.ListOfVariableIndices())
    num_var = length(variables)
    x_l = fill(-Inf, num_var)
    x_u = fill(Inf, num_var)
    x_cat = fill(:Cont, num_var)

    variable_map = Dict{MOI.VariableIndex, Int}()
    for (i, variable) in enumerate(variables)
        variable_map[variable] = i
    end

    for set_type in (MOI.LessThan{Float64}, MOI.GreaterThan{Float64})
        for c_ref in MOI.get(model,
            MOI.ListOfConstraintIndices{MOI.SingleVariable, set_type}())
            c_func = MOI.get(model, MOI.ConstraintFunction(), c_ref)
            c_set = MOI.get(model, MOI.ConstraintSet(), c_ref)
            v_index = variable_map[c_func.variable]
            lower, upper = set_to_bounds(c_set)
            x_l[v_index] = lower
            x_u[v_index] = upper
        end
    end

    for set_type in (MOI.ZeroOne, MOI.Integer)
        for c_ref in MOI.get(model,
            MOI.ListOfConstraintIndices{MOI.SingleVariable, set_type}())
            c_func = MOI.get(model, MOI.ConstraintFunction(), c_ref)
            c_set = MOI.get(model, MOI.ConstraintSet(), c_ref)
            v_index = variable_map[c_func.variable]
            x_cat[v_index] = set_to_cat(c_set)
        end
    end

    # Get the optimzation sense.
    opt_sense = MOI.get(model, MOI.OptimizationSense())
    sense = opt_sense == MOI.MAX_SENSE ? :Max : :Min

    nlp_block = try
        MOI.get(model, MOI.NLPBlock())
    catch ex
        error("Expected a NLPBLock.")
    end

    # Extract constraint bounds.
    num_con = length(nlp_block.constraint_bounds)
    g_l = fill(-Inf, num_con)
    g_u = fill(Inf, num_con)
    for (i, bound) in enumerate(nlp_block.constraint_bounds)
        g_l[i] = bound.lower
        g_u[i] = bound.upper
    end

    mpb_model = MPB.NonlinearModel(mpb_solver)
    MPB.loadproblem!(
        mpb_model, num_var, num_con, x_l, x_u, g_l, g_u, sense,
        nlp_block.evaluator
    )

    MPB.setvartype!(mpb_model, x_cat)

    MPB.optimize!(mpb_model)

    # Handle statuses.
    status = MPB.status(mpb_model)
    termination_status = MOI.OTHER_ERROR
    primal_status = MOI.NO_SOLUTION
    dual_status = MOI.NO_SOLUTION
    if status == :Optimal
        termination_status = MOI.OPTIMAL
        primal_status = MOI.FEASIBLE_POINT
    elseif status == :Infeasible
        termination_status = MOI.INFEASIBLE
    elseif status == :Unbounded
        termination_status = MOI.DUAL_INFEASIBLE
    elseif status == :UserLimit
        termination_status = MOI.OTHER_LIMIT
    elseif status == :Error
        termination_status = MOI.OTHER_ERROR
    end
    MOI.set(model, MOI.TerminationStatus(), termination_status)
    MOI.set(model, MOI.PrimalStatus(), primal_status)
    MOI.set(model, MOI.DualStatus(), dual_status)

    x_solution = MPB.getsolution(mpb_model)
    for (variable, sol) in zip(variables, x_solution)
        MOI.get(model, MOI.VariablePrimal(), variable, sol)
    end

    obj_value = MPB.getobjval(model)
    MOI.set(model, MOI.ObjectiveValue(), obj_value)

    return
end

end
