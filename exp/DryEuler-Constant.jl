include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

import PyPlot


function constant_test(vertical_method::String, t_end::Float64 = 100.0, Nz::Int64=32)
    
    Np = 4
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)
    topology_type = "AtmoLES"
    
    
    
    Nx = 2
    Lx, Lz = 20.0e3, 30.0e3
    
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    viscous, ν, Pr = false, NaN64, NaN64
    gravity = false
    hydrostatic_balance = false
    
    
    
    num_state_prognostic = 4
    

    app = DryAtmo("no-penetration", nothing, "no-penetration", nothing,  "periodic", nothing, "periodic", nothing, 
    viscous, ν, Pr, gravity, hydrostatic_balance)
    # update_sponge_params!(app, -1.0, Lz, Lz*1/2.0)

    cfl = 0.8
    dt0 = min(cfl/Np^2*Lx/Nx/330, cfl*Lz/Nz/330)
    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.8, "dt0" => dt0, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    
    # set initial condition 
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    state_prognostic_0 = zeros(Nl, num_state_prognostic, nelem)
    state_prognostic_0[:, 1, :] .= 1.0
    state_prognostic_0[:, 4, :] .= 10000.0

    set_init_state!(solver, state_prognostic_0)


    Q = solve!(solver)

    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    ρ  = reshape(state_primitive[:, 1 ,:], (Nl * Nx, Nz))
    u  = reshape(state_primitive[:, 2 ,:], (Nl * Nx, Nz))
    w  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz))
    p  = reshape(state_primitive[:, 4 ,:], (Nl * Nx, Nz))

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
    fig.savefig("Hydrostatic_Balance"*vertical_method*"_x.png")



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
    fig.savefig("Hydrostatic_Balance"*vertical_method*"_z.png")
    
end

t_end = 86400.0 
Nz = 32
constant_test("FV",    t_end,  Nz)
#hydrostatic_balance("WENO3", t_end,  Nz)
#hydrostatic_balance("WENO5", t_end,  Nz)
