include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot


function init_risingbubble(x::Float64, z::Float64)
    # Problem float-type
    gas_constant() =     8.3144598
    molmass_dryair() = 28.97e-3
    kappa_d()        = 2 / 7
    # Unpack constant parameters
    R_gas::Float64 = gas_constant() / molmass_dryair()
    c_p::Float64 = R_gas / kappa_d()
    c_v::Float64 = c_p - R_gas
    p0::Float64 = 1.01325e5
    _grav::Float64 = 9.8

    γ::Float64 = c_p / c_v


    xc, zc = 0.0, 2000.0
    r = sqrt((x - xc)^2 + (z - zc)^2)
    rc = 2000
    θamplitude = 2
    T_surface = 300.0
    θ_ref = T_surface
    Δθ = 0
    if r <= rc
        Δθ = θamplitude * (1.0 - r / rc)
    end

    # Compute perturbed thermodynamic state:
    θ = θ_ref + Δθ                                      ## potential temperature
    π_exner = 1.0 - _grav / (c_p * θ) * z             ## exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas)      ## density
    T = θ * π_exner
    p = ρ * R_gas * T
    
    ρu = [0.0; 0.0]                   ## momentum
    
    ρe = p/(γ-1) + ρ*_grav*z
                
    return [ρ; ρu; ρe]

end


function rising_bubble(vertical_method::String, Np::Int64=2, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    
    Nl = Np+1
    topology_type = "AtmoLES"
    
    
    Nx, Nz = 80,   80*Nl
    Lx, Lz = 10000.0, 10000.0
    
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    # viscous, ν, Pr = false, NaN64, NaN64
    viscous, ν, Pr = true, 0.01, 0.72
    gravity = true
    hydrostatic_balance = true
    
    # set initial condition 
    num_state_prognostic, nelem = 4, Nx*Nz
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    
    
    app = DryAtmo("no-penetration", [0.0;0.0;0.0], "no-penetration", [0.0;0.0;0.0],  "periodic", nothing, "periodic", nothing, 
    viscous, ν, Pr,  
    gravity, hydrostatic_balance)
    
    
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.5/Np, "dt0" => 0.02, "t_end" => 0.050000, "vertical_method" => vertical_method)
    
    solver = Solver(app, mesh, params)
    


    init_state!(app, mesh, state_prognostic_0, init_risingbubble)
    set_init_state!(solver, state_prognostic_0)

    
    
    

    state_primitive_0 = similar(state_prognostic_0)
    prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive_0)
    visual(mesh, state_primitive_0[:,1,:], "Rising_Bubble_rho_init.png")
    visual(mesh, state_primitive_0[:,2,:], "Rising_Bubble_u_init.png")
    visual(mesh, state_primitive_0[:,3,:], "Rising_Bubble_v_init.png")
    visual(mesh, state_primitive_0[:,4,:], "Rising_Bubble_T_init.png")
    
    
    Q = solve!(solver)
    
    ρ  = reshape(Q[:, 1 ,:], (Nl * Nx, Nz)) 
    nx_plot, nz_plot = div(Nl * Nx, 2) , div(Nz, 2)
    fig, (ax1, ax2) = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    xx = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, nz_plot]
    zz = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[nx_plot, :]
    
    ax1.plot(xx, ρ[:, nz_plot],  "-o", fillstyle = "none", label = "xx")
    ax2.plot(zz, ρ[nx_plot, :],  "-o", fillstyle = "none", label = "zz")
    ax1.set_ylabel("ρ")
    fig.savefig("Rising_Bubble_rho_xz.png")
    PyPlot.close(fig)


   
    



    

    
    state_primitive = similar(Q)
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    visual(mesh, state_primitive[:,1,:], "Rising_Bubble_rho_end.png")
    visual(mesh, state_primitive[:,2,:], "Rising_Bubble_u_end.png")
    visual(mesh, state_primitive[:,3,:], "Rising_Bubble_v_end.png")
    visual(mesh, state_primitive[:,4,:], "Rising_Bubble_p_end.png")
    
    
end


vertical_method = "FV"
Np = 3
rising_bubble(vertical_method, Np)

