include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot


function hydrostatic_balance(vertical_method::String, t_end::Float64 = 100.0, Nz::Int64=32)
    
    Np = 3
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    topology_type = "AtmoLES"
    
    
    
    Nx = 16
    Lx, Lz = 16.0e3, 8.0e3
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mountain_wrap_les!(Nl, Nx, Nz, Lx, Lz, topology)

    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    gravity = true
    
    
    
    
    num_state_prognostic = 4
    
    app = DryEuler("no-penetration", nothing, "no-penetration", zeros(Float64, num_state_prognostic),  "periodic", nothing, "periodic", nothing, gravity)
    update_sponge_params!(app, -1.0, Lz, Lz*1/2.0)
        
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.8/Np, "dt0" => 10.0, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    

    # set initial condition
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    T_virt_surf, T_min_ref, H_t = 280.0, 230.0, 9.0e3
    profile_0 = init_hydrostatic_balance!(app,  mesh,  state_prognostic_0, solver.state_auxiliary_vol_l,  T_virt_surf, T_min_ref, H_t)
    set_init_state!(solver, state_prognostic_0)


    # update reference state
    # T_virt_surf, T_min_ref, H_t =  290.0, 220.0, 8.0e3
    # state_prognostic_ref = ones(Nl, num_state_prognostic, nelem)
    # profile_ref = init_hydrostatic_balance!(app,  mesh,  state_prognostic_ref, solver.state_auxiliary_vol_l,  T_virt_surf, T_min_ref, H_t)
    # state_auxiliary_vol_l  =  solver.state_auxiliary_vol_l     
    # state_auxiliary_vol_q  =  solver.state_auxiliary_vol_q   
    # state_auxiliary_surf_h =  solver.state_auxiliary_surf_h   
    # state_auxiliary_surf_v =  solver.state_auxiliary_surf_v
    # state_primitive_ref = similar(state_prognostic_ref)
    # prog_to_prim!(app, state_prognostic_ref, solver.state_auxiliary_vol_l, state_primitive_ref)
    # update_state_auxiliary!(app, mesh, state_primitive_ref, state_auxiliary_vol_l, state_auxiliary_vol_q, state_auxiliary_surf_h, state_auxiliary_surf_v)



    # visualize
    visual(mesh, state_prognostic_0[:,1,:], "Hydrostatic_Balance_Mountain_init_"*vertical_method*".png")

    # solve
    Q = solve!(solver)
    
    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    ρ  = reshape(state_primitive[:, 1 ,:], (Nl * Nx, Nz))
    u  = reshape(state_primitive[:, 2 ,:], (Nl * Nx, Nz))
    w  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz))
    p  = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz))

    visual(mesh, ρ, "Hydrostatic_Balance_Mountain_rho_"*vertical_method*".png")
    visual(mesh, u, "Hydrostatic_Balance_Mountain_u_"*vertical_method*".png")
    visual(mesh, w, "Hydrostatic_Balance_Mountain_w_"*vertical_method*".png")
    visual(mesh, p, "Hydrostatic_Balance_Mountain_p_"*vertical_method*".png")

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
    fig.savefig("Hydrostatic_Balance_Mountain"*vertical_method*"_x.png")



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
    fig.savefig("Hydrostatic_Balance_Mountain"*vertical_method*"_z.png")
    
end

t_end = 86400.0 / 2.0
Nz = 32
hydrostatic_balance("FV",    t_end,  Nz)
# hydrostatic_balance("WENO3", t_end,  Nz)
# hydrostatic_balance("WENO5", t_end,  Nz)


