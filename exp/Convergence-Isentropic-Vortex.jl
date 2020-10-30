include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot

function integrate_l2(ρ::Array{Float64, 2}, Δx, Δz, ωl)
    e2 = 0.0
    Nl, nelem = size(ρ)
    for il = 1:Nl
        for e = 1:nelem
            e2 += ρ[il, e]^2 * ωl[il] 
        end
    end
    return sqrt(e2 * Δx * Δz/2.0)
end

function isentropic_vortex(level::Int64, vertical_method::String, Np::Int64=2, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    
    Nl = Np+1
    topology_type = "AtmoLES"
    
    
    Nx = 2^(level-1)*5
    Nz = Nx * Nl
    Lx, Lz = 10.0, 10.0
    
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    viscous, ν, Pr = false, NaN64, NaN64
    # viscous, ν, Pr = true, 0.01, 0.72
    gravity = false
    hydrostatic_balance = false
    
    # set initial condition 
    num_state_prognostic, nelem = 4, Nx*Nz
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    
    
    app = DryAtmo("periodic", nothing, "periodic", nothing,  "periodic", nothing, "periodic", nothing, 
    viscous, ν, Pr,  
    gravity, hydrostatic_balance)
    
    t_end = 5.0
    dt0 = 0.02/2^(level-1)
    params = Dict("time_integrator" => "RK4", "cfl_freqency" => -1, "cfl" => 0.5, "dt0" => dt0, "t_end" => t_end, "vertical_method" => vertical_method)
    
    solver = Solver(app, mesh, params)
    
    
    function init_func(x::Float64, z::Float64, t::Float64 = 0.0)
        
        γ = 1.4  
        ρ∞ = 1.0
        
        u∞ = 0.5
        M∞ = 0.5
        c∞ = u∞/M∞ 
        θ = π/4
        p∞ = ρ∞*c∞^2/γ
        ū, v̄ = u∞*cos(θ), u∞*sin(θ)
        ϵ = 5.0
        r = 1.0
        x₀, z₀ = 0.0, 5.0
        
        
        f(x, z, t) = (1 - (x - x₀ - ū*t)^2 - (z - z₀ - v̄*t)^2)/r^2
        u = u∞*(cos(θ) - ϵ*((z -z₀) - v̄*t)/(2*π*r)*exp(f(x,z,t)/2.0))
        v = u∞*(sin(θ) + ϵ*((x -x₀) - ū*t)/(2*π*r)*exp(f(x,z,t)/2.0))
        ρ = ρ∞*(1 - ϵ^2*(γ-1)*M∞^2/(8.0*π^2)*exp(f(x,z, t)))^(1/(γ-1))
        p = p∞*(1 - ϵ^2*(γ-1)*M∞^2/(8.0*π^2)*exp(f(x,z, t)))^(γ/(γ-1))
        
        ρe = p/(γ-1) + 0.5*(ρ*u^2 + ρ*v^2) 
        return [ρ; ρ*u; ρ*v; ρe]
    end
    
    init_state!(app, mesh, state_prognostic_0, init_func)
    set_init_state!(solver, state_prognostic_0)
    
    ref_sol_func = (x, z)->init_func(x, z, t_end)
    state_prognostic_ref = similar(state_prognostic_0)
    init_state!(app, mesh, state_prognostic_ref, ref_sol_func)
    
    
    state_primitive_0 = similar(state_prognostic_0)
    prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive_0)
    ρ0 = state_primitive_0[:, 1, :]
    
    
    state_primitive_ref = similar(state_prognostic_ref)
    prog_to_prim!(app, state_prognostic_ref, solver.state_auxiliary_vol_l, state_primitive_ref)
    ρref = state_primitive_ref[:, 1, :]
    
    Q = solve!(solver)
    state_primitive = similar(Q)
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    ρ = state_primitive[:, 1, :]
    
    
    Δx, Δz = Lx/Nx, Lz/Nz
    ωl = mesh.ωl
    e2 = integrate_l2(ρ - ρref, Δx, Δz, ωl)
    
    visual(mesh, ρ, "Isentropic_Vortex_ρ.png")
    visual(mesh, ρref, "Isentropic_Vortex_ρ_ref.png")
    
    return e2
end

function error_plot()
    """
    [ Info: ("Np: ", 3, " vertical_method : ", "WENO5")
    ("level= ", 1, " L2 error is ", 0.02824643048689777)
    ("level= ", 2, " L2 error is ", 0.0032442912324973177)
    ("level= ", 3, " L2 error is ", 0.00038466859288068683)
    ("level= ", 4, " L2 error is ", 8.51235578640324e-5)
    
    [ Info: ("Np: ", 4, " vertical_method : ", "WENO5")
    ("level= ", 1, " L2 error is ", 0.012621293981719675)
    ("level= ", 2, " L2 error is ", 0.0009930168720410414)
    ("level= ", 3, " L2 error is ", 0.00021660221955972485)
    ("level= ", 4, " L2 error is ", 5.380180031947584e-5)
    
    [ Info: ("Np: ", 3, " vertical_method : ", "FV")
    ("level= ", 1, " L2 error is ", 0.05699014170090471)
    ("level= ", 2, " L2 error is ", 0.013054692847686582)
    ("level= ", 3, " L2 error is ", 0.0026915439443542222)
    ("level= ", 4, " L2 error is ", 0.0005747335233563602)
    
    [ Info: ("Np: ", 4, " vertical_method : ", "FV")
    ("level= ", 1, " L2 error is ", 0.0351777767401445)
    ("level= ", 2, " L2 error is ", 0.007528094976291764)
    ("level= ", 3, " L2 error is ", 0.0015967225442441817)
    ("level= ", 4, " L2 error is ", 0.0003527731779933969)
    """
    
    labels = ["p3-WENO5" "p4-WENO5" "p3-FV" "p4-FV" ]
    Δxs =    [10.0/5 ; 10.0/(2*5) ; 10.0/(2^2*5); 10.0/(2^3*5)]
    errors = [0.02824643048689777 0.0032442912324973177 0.00038466859288068683 8.51235578640324e-5; 
    0.012621293981719675 0.0009930168720410414 0.00021660221955972485 5.380180031947584e-5; 
    0.05699014170090471  0.013054692847686582 0.0026915439443542222 0.0005747335233563602;
    0.0351777767401445 0.007528094976291764 0.0015967225442441817 0.0003527731779933969]
    
    for i = 1: 4
        PyPlot.loglog(Δxs, errors[i, :], "-o", label=labels[i], base=10)
    end


    PyPlot.loglog(Δxs, 0.01*Δxs.^2, "--", label="Δx²", base=10)
    PyPlot.loglog(Δxs, 0.01*Δxs.^3, "--", label="Δx³", base=10)
    
    
    PyPlot.xlabel("Δx")
    PyPlot.ylabel("L₂ error")
    PyPlot.legend()
    PyPlot.grid()
    PyPlot.title("Isentropic_Vortex (Isentropic mesh RK4)")
    
end




# for vertical_method in ["WENO5"; "FV"]
#     for Np in [3; 4]
#         @info "Np: ", Np, " vertical_method : ", vertical_method
#         for level = 1:4
#             e2 = isentropic_vortex(level, vertical_method, Np)
#             @info "level= ",  level, " L2 error is ", e2
#         end
#     end
# end


error_plot()