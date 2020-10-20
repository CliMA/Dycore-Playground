include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot

function init_agnesi_hs_lin(z::Float64)
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

    c::Float64 = c_v / R_gas
    c2::Float64 = R_gas / c_p

    # Define initial thermal field as isothermal
    Tiso::Float64 = 280.0
    θ0::Float64 = Tiso

    # Assign a value to the Brunt-Vaisala frquencey:
    Brunt::Float64 = 0.01
    Brunt2::Float64 = Brunt * Brunt
    g2::Float64 = _grav * _grav

    # π_exner = (p/p0)^{R/cp}
    π_exner::Float64 = 1 + _grav^2/(c_p*θ0*Brunt2) * (exp(-Brunt2*z/_grav) - 1)
    θ::Float64 = θ0 * exp(Brunt2 * z / _grav)
    ρ::Float64 = p0 / (R_gas * θ) * (π_exner)^c
    

    # Compute perturbed thermodynamic state:
    T = θ * π_exner
    p = ρ * R_gas * T
    
    p_test = p0*π_exner^(c_p/R_gas)

    # @info "same : ", p, p_test



    # @info "same: ", ρ, p0/θ/R_gas*(p/p0)^(c_v/c_p)
    # @info _grav^2, c_p, c_p*θ0*Brunt2, exp(-z*Brunt2/_grav)
    # @info (1 + _grav^2/(c_p*θ0*Brunt2)*(exp(-z*Brunt2/_grav) - 1))
    # @info "pp:", p, p0*(1 + _grav^2/(c_p*θ0*Brunt2)*(exp(-z*Brunt2/_grav) - 1))^(c_p/R_gas)

    return p, ρ

end





function hydrostatic_balance(vertical_method::String, t_end::Float64 = 100.0, Nz::Int64=32)
    
    Np = 4
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    topology_type = "AtmoLES"
    
    
    
    Nx = 50
    Lx, Lz =  50.0e3, 21.0e3 # 144.0e3, 30.0e3
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mountain_wrap_les!(Nl, Nx, Nz, Lx, Lz, topology)

    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    gravity = true
    hydrostatic_balance = true
    
    
    
    num_state_prognostic = 4
    u_init = [10.0;0.0]
    app = DryEuler("no-penetration", nothing, "no-penetration", zeros(Float64, num_state_prognostic),  "periodic", nothing, "periodic", nothing, gravity, hydrostatic_balance)
    update_sponge_params!(app, -1.0, Lz, 10e3, Lx/2.0, 15e3, u_init)
        
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 1.0, "dt0" => 10.0, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    

    # set initial condition
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    state_auxiliary_vol_l = solver.state_auxiliary_vol_l
    
    init_discrete_hydrostatic_balance!(app, mesh, state_prognostic_0, state_auxiliary_vol_l, init_agnesi_hs_lin, u_init)
    set_init_state!(solver, state_prognostic_0)


 

    # visualize
    visual(mesh, state_prognostic_0[:,1,:], "NonHydrostatic_Balance_Mountain_init_"*vertical_method*".png")

    # solve
    Q = solve!(solver)
    
    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    ρ  = reshape(state_primitive[:, 1 ,:], (Nl * Nx, Nz))
    u  = reshape(state_primitive[:, 2 ,:], (Nl * Nx, Nz))
    w  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz))
    p  = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz))

    visual(mesh, ρ, "NonHydrostatic_Balance_Mountain_rho_"*vertical_method*".png")
    visual(mesh, u, "NonHydrostatic_Balance_Mountain_u_"*vertical_method*".png")
    visual(mesh, w, "NonHydrostatic_Balance_Mountain_w_"*vertical_method*".png")
    visual(mesh, p, "NonHydrostatic_Balance_Mountain_p_"*vertical_method*".png")

    state_primitive_0 = copy(solver.state_primitive)
    prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive_0)
    ρ0  = reshape(state_primitive_0[:, 1 ,:], (Nl * Nx, Nz))
    u0  = reshape(state_primitive_0[:, 2 ,:], (Nl * Nx, Nz))
    w0  = reshape(state_primitive_0[:, 3 ,:], (Nl * Nx, Nz))
    p0  = reshape(state_primitive_0[:, 4 ,:], (Nl * Nx, Nz))


    nx_plot = div(Nl * Nx, 2)
    zz = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[nx_plot, :]
    fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=true, figsize=(12,6))
    ax1.plot(ρ0[nx_plot, :], zz, "-o", fillstyle = "none", label = "Init")
    ax1.plot(ρ[nx_plot, :], zz, "-", fillstyle = "none", label = vertical_method)
    ax1.legend()
    ax1.set_xlabel("ρ")
    ax2.plot(sqrt.(u0.^2 + w0.^2)[nx_plot, :], zz, "-o", fillstyle = "none", label = "Init")
    ax2.plot(sqrt.(u.^2 + w.^2)[nx_plot, :], zz, "-", fillstyle = "none", label = vertical_method)
    ax2.legend()
    ax2.set_xlabel("|v|")
    ax3.plot(p0[nx_plot, :], zz, "-o", fillstyle = "none", label = "Init")
    ax3.plot(p[nx_plot, :], zz, "-", fillstyle = "none", label = vertical_method)
    ax3.legend()
    ax3.set_xlabel("p")
    fig.savefig("NonHydrostatic_Balance_Mountain"*vertical_method*"_x.png")



    nz_plot = div(Nz, 2)
    xx = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, nz_plot]
    fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    ax1.plot(xx, ρ0[:, nz_plot], "-o", fillstyle = "none", label = "Init")
    ax1.plot(xx, ρ[:, nz_plot], "-", fillstyle = "none", label = vertical_method)
    ax1.legend()
    ax1.set_ylabel("ρ")
    ax2.plot(xx, sqrt.(u0.^2 + w0.^2)[:, nz_plot], "-o", fillstyle = "none", label = "Init")
    ax2.plot(xx, sqrt.(u.^2 + w.^2)[:, nz_plot],  "-", fillstyle = "none", label = vertical_method)
    ax2.legend()
    ax2.set_ylabel("|v|")
    ax3.plot(xx, p0[:, nz_plot],  "-o", fillstyle = "none", label = "Init")
    ax3.plot(xx, p[:, nz_plot], "-", fillstyle = "none", label = vertical_method)
    ax3.legend()
    ax3.set_ylabel("p")
    fig.savefig("NonHydrostatic_Balance_Mountain"*vertical_method*"_z.png")
    
end

t_end = 3600.0 #86400.0 / 2.0
Nz = 100
# hydrostatic_balance("FV",    t_end,  Nz)
# hydrostatic_balance("WENO3", t_end,  Nz)
hydrostatic_balance("WENO5", t_end,  Nz)


