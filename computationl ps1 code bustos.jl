cd("C:\\Users\\marti\\OneDrive - The University of Western Ontario\\Documents\\Mar\\Académico\\PhD (UWO)\\02) Second year\\02) Winter\\Computational Macro\\Ps1 trial")
mkpath("Figures")
ENV["GKS_ENCODING"]="utf-8"
using Pkg; Pkg.add(["Parameters", "Plots", "UnPack", "LinearAlgebra", "Interpolations"])
using Parameters
using UnPack
using LinearAlgebra
using Plots
using Interpolations
#Pkg.update("Parameters")
gr()


# Parameters 
struct Par
    # Model Parameters
    z::Float64
    α::Float64
    β::Float64
    # VFI Parameters
    max_iter::Int64
    dist_tol::Float64
end

par = Par(1, 1/3, 0.98, 10000, 1E-2)


# Utility function
function utility(k, k_prime, par::Par)
    z, α = par.z, par.α
    c = z * (k^α) - k_prime
    if c > 0
        return log(c)
    else
        return -Inf
    end
end

# Steady state
function steady_state(z, par::Par)
    α, β = par.α, par.β
    k = (α * β * z) ^ (1 / (1 - α))
    y = z * (k ^ α)
    c = y - k
    r = α  * (y / k)
    w = (1 - α ) * y
    return k, y, c, r, w
end


# Value function (analytical solution)
function V_analytical(k, par::Par)
    z, α, β = par.z, par.α, par.β
    V = (1 / (1 - β)) * (log.(z ./ (1 - α * β)) .+ (α * β ./ (1 -α*β)) .* log.(z * α * β)) .+ (α ./ (1 - α * β)) .* log.(k)
    return V
end


# Policy function (analytical solution)
function policy_function_analytical(k, par::Par)
    @unpack z, α, β = par
    k_prime = α * β * z * (k^α)
    return k_prime
end


# Grid for k
function Make_K_Grid(n_k, k, k_shocks)
    k_min = min(k, minimum(k_shocks)) * 0.5
    k_max = max(k, maximum(k_shocks)) * 1.5
    k_grid = range(k_min, k_max; length=n_k)
    return k_grid
end


#function T_grid_loop(V_old, k_grid, k_, par::Par)
#    n_k = length(k_grid)
#    V_new = zeros(n_k)
#    for i_k in 1:n_k
#        k = k_grid[i_k]
#        V = [utility(k, k_prime, par) + par.β * V_old[i_k_prime] for (i_k_prime, k_prime) in enumerate(k_grid)]
#        V_max, k_max_ind = findmax(V)
#        V_new[i_k] = V_max
#        k_[i_k] = k_grid[k_max_ind]
#    end
#    return V_new, k_
#end


# Value function iteration
#function VFI(par::Par, k, k_shocks)
    # Initialize iteration counter
#    iter = 0
    # Calculate grid for k
#    n_k = 1000
#    k_grid = Make_K_Grid(n_k, k, k_shocks)
    # Initialize V_old and V_new
#    V_old = zeros(n_k)
#    V_new = V_analytical(k_grid, par)
    # Initialize k_ as an empty array with the same length as k_grid
#    k_ = Array{Float64,1}(undef, n_k)
    # Calculate distance between V_old and V_new
#    dist = norm(V_old - V_new, Inf)
    # Iterate until distance < dist_tol or max_iter is reached
#    while dist > par.dist_tol && iter < par.max_iter
#        iter += 1
#        V_old = V_new
#        V_new, k_ = T_grid_loop(V_old, k_grid, k_, par)
#        dist = norm(V_old - V_new, Inf)
#    end
#    return V_new, k_, k_grid
#end


# Simulate shocks
function simulate_variables(policy_function, shock_k, shock_z, n_periods, par, ss_values)
    results = Dict()
    k_sim = Array{Float64}(undef, n_periods)
    k_sim[1] = shock_k
    epsilon = 1E-2  
    n_last = 100  
    for t in 2:n_periods
        k_sim[t] = policy_function(k_sim[t-1], par)
        if t > n_last && check_convergence(ss_values[1], k_sim[t-n_last:t], epsilon, n_last)
            resize!(k_sim, t)
            break
        end
    end
    y_sim = shock_z * k_sim .^ par.α
    c_sim = y_sim[1:end-1] - k_sim[2:end] 
    w_sim = (1-par.α) * shock_z * k_sim.^(par.α)
    r_sim = par.α * shock_z * k_sim.^(par.α-1)
    results = (k_sim, c_sim, y_sim, w_sim, r_sim)
    return results
end


function simulate_shocks(policy_function, shock_k, shock_z, n_periods, par)
    k_sim = Array{Float64}(undef, n_periods)
    k_sim[1] = shock_k
    for t in 2:n_periods
        k_sim[t] = policy_function(k_sim[t-1], par)
    end
    return k_sim
end


function check_convergence(ss_value::Float64, sim_var::Vector{Float64}, epsilon::Float64, n_last::Int64)
    last_values = sim_var[end-n_last+1:end] 
    converged = all(abs.(last_values .- ss_value) .< epsilon)  
    return converged
end


function main()
    par = Par(1, 1/3, 0.98, 10000, 1E-2)

    n_periods = 10000
    shock_z = [1.0, 1.05]

    simulation_results = []

    for (i, z) in enumerate(shock_z)
        par_shocked = Par(z, par.α, par.β, par.max_iter, par.dist_tol)
        ss_values = steady_state(z, par_shocked)
        shock_k = ss_values[1] * (i == 2 ? 0.8 : 1.0)
        policy_function = policy_function_analytical
        results = simulate_variables(policy_function, shock_k, z, n_periods, par, ss_values)
        push!(simulation_results, results)
    end

    labels = ["Capital", "Consumption", "Output", "Wage", "Real Interest Rate"]
    shock_types = ["Capital Shock", "Productivity Shock"]
    
    for (i, results) in enumerate(simulation_results)
        for (j, label) in enumerate(labels) 
            sim_var = results[j]
            ss_value = steady_state(shock_z[i], par)[j]
            p = plot(title = "$(label) - $(shock_types[i])")

            epsilon = 1E-2  
            n_last = 100
            # Get the period of convergence
            period_to_convergence = findfirst(x -> check_convergence(ss_value, sim_var[x-n_last+1:x], epsilon, n_last), n_last:length(sim_var))
            # If convergence was not achieved within n_periods, set the limit to n_periods
            if period_to_convergence == nothing
                period_to_convergence = n_periods
            end
            xlims!(p, 1, period_to_convergence)
            plot!(p, 1:length(sim_var), sim_var, label = "$(label) New Steady State", linewidth = 2, legend = :bottomright)
            hline!(p, [steady_state(1, par)[j]], color = :red, linestyle = :dash, label = "$(label) Initial Steady State")
            hline!(p, [ss_value], color = :blue, linestyle = :dash, label = "$(label) New Steady State")
            xlabel!(p, "Periods")
            ylabel!(p, label)
            savefig(p, "Figures/$(label)_$(shock_types[i]).png")
        end
    end      
end


main()