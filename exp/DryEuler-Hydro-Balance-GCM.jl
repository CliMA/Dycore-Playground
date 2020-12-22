include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot


function hydrostatic_balance(vertical_method::String, t_end::Float64 = 100.0, Nz::Int64=32)
    
    Np = 3
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    topology_type = "AtmoGCM"
    
    
    
    Nx = 32
    r = 6371e3
    R = r + 30e3
    
    
    topology_size = [r; R]
    topology = topology_gcm(Nl, Nx, Nz, r, R)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    viscous, ν, Pr = false, NaN64, NaN64
    gravity = true
    hydrostatic_balance = true
    
    
    
    num_state_prognostic = 4
    
    app = DryAtmo("no-penetration", nothing, "no-penetration", zeros(Float64, num_state_prognostic),  "periodic", nothing, "periodic", nothing, viscous, ν, Pr, gravity, hydrostatic_balance)
    update_sponge_params!(app, -1.0, R-r, (R-r)*1/2.0, Inf64, Inf64, [0.0,0.0])
    
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.4, "dt0" => 10.0, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    
    # set initial condition 
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz 
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    T_virt_surf, T_min_ref, H_t = 280.0, 230.0, 9.0e3
    profile = init_discrete_hydrostatic_balance!(app,  mesh,  state_prognostic_0, solver.state_auxiliary_vol_l,  T_virt_surf, T_min_ref, H_t)
    set_init_state!(solver, state_prognostic_0)


 
    Q = solve!(solver)





    xx, zz = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[1, :], reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[1, :]
    alt = sqrt.(xx.^2 + zz.^2) .- r
    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    ρ  = reshape(state_primitive[:, 1 ,:], (Nl * Nx, Nz)) 
    u  = reshape(state_primitive[:, 2 ,:], (Nl * Nx, Nz)) 
    w  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz)) 
    p  = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz)) 

    state_primitive_0 = solver.state_primitive
    prog_to_prim!(app, state_prognostic_0, solver.state_auxiliary_vol_l, state_primitive_0)
    ρ0  = reshape(state_primitive_0[:, 1 ,:], (Nl * Nx, Nz)) 
    u0  = reshape(state_primitive_0[:, 2 ,:], (Nl * Nx, Nz)) 
    w0  = reshape(state_primitive_0[:, 3 ,:], (Nl * Nx, Nz)) 
    p0  = reshape(state_primitive_0[:, 4 ,:], (Nl * Nx, Nz))



    fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=true, figsize=(12,6))
    ax1.plot(ρ0[1, :], alt, "-o", fillstyle = "none", label = "Ref")
    ax1.plot(ρ[1, :], alt, "-", fillstyle = "none", label = vertical_method)
    ax1.legend()
    ax1.set_xlabel("ρ")
    ax2.plot(sqrt.(u0[1, :].^2 + w0[1, :].^2), alt, "-o", fillstyle = "none", label = "Ref")
    ax2.plot(sqrt.(u[1, :].^2 + w[1, :].^2), alt, "-", fillstyle = "none", label = vertical_method)
    ax2.legend()
    ax2.set_xlabel("|v|")
    ax3.plot(p0[1, :], alt, "-o", fillstyle = "none", label = "Ref")
    ax3.plot(p[1, :], alt, "-", fillstyle = "none", label = vertical_method)
    ax3.legend()
    ax3.set_xlabel("p")
    fig.savefig("Hydrostatic_Balance_GCM"*vertical_method*".png")



    nz_plot = div(Nz, 2)
    xx = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, nz_plot]
    fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    ax1.plot(xx, ρ0[:, nz_plot], "-o", fillstyle = "none", label = "Init")
    ax1.plot(xx, ρ[:, nz_plot], "-", fillstyle = "none", label = vertical_method)
    ax1.legend()
    ax1.set_ylabel("ρ")
    ax2.plot(xx, sqrt.(u0.^2 + w0.^2)[:, nz_plot], "-o", fillstyle = "none", label = "Init")
    ax2.plot(xx, sqrt.(u.^2 + w.^2)[:, nz_plot],  "-", fillstyle = "none", label = vertical_method)
    # ax2.plot(xx, sqrt.(u0.^2 )[:, nz_plot], "-o", fillstyle = "none", label = "Init")
    # ax2.plot(xx, sqrt.(u.^2 )[:, nz_plot],  "-", fillstyle = "none", label = vertical_method)
    ax2.legend()
    ax2.set_ylabel("|v|")
    ax3.plot(xx, p0[:, nz_plot],  "-o", fillstyle = "none", label = "Init")
    ax3.plot(xx, p[:, nz_plot], "-", fillstyle = "none", label = vertical_method)
    ax3.legend()
    ax3.set_ylabel("p")
    fig.savefig("Hydrostatic_Balance_GCM"*vertical_method*"_z.png")
    
    
end

t_end =  86400.0 
Nz = 32
hydrostatic_balance("FV",    t_end,  Nz)
# hydrostatic_balance("WENO3", t_end,  Nz)
hydrostatic_balance("WENO5", t_end,  Nz)




