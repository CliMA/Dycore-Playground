include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")




function shock_tube(direction::String, vertical_method::String, Np::Int64=2, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    @assert(direction == "horizontal" || direction == "vertical")
    

    Nl = Np+1
    topology_type = "AtmoLES"
    
    if direction == "vertical"
        
        Nx, Nz = 2,   100
        Lx, Lz = 1.0, 1.0

    else direction == "horizontal"

        Nx, Nz = 50,  2
        Lx, Lz = 1.0, 1.0
        
    end
    
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    gravity = false
    hydrostatic_balance = false

    # set initial condition 
    num_state_prognostic, nelem = 4, Nx*Nz
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    prim_l = [1.0;   0.0; 0.0; 1.0]
    prim_r = [0.125; 0.0; 0.0; 0.1]

    

    
    if direction == "vertical"
        app = DryEuler("no-slip", nothing, "outlet", prim_r,  "periodic", nothing, "periodic", nothing, gravity, hydrostatic_balance)
    else direction == "horizontal"
        app = DryEuler("periodic", nothing, "periodic", nothing, "no-slip", nothing, "no-slip", nothing,  gravity, hydrostatic_balance)
    end
    

    params = Dict("time_integrator" => "RK2", "cfl_freqency" => -1, "cfl" => 0.8/Np, "dt0" => 0.02, "t_end" => 0.2000, "vertical_method" => vertical_method)

    solver = Solver(app, mesh, params)
    
    
    
    cons_l = prim_to_prog(app, prim_l, zeros(Float64, app.num_state_auxiliary))
    cons_r = prim_to_prog(app, prim_r, zeros(Float64, app.num_state_auxiliary))

    
    function shock_tube_func(x::Float64, z::Float64)
        if direction == "vertical"
            if z  <= Lz/2.0
                return cons_l
            else
                return cons_r
            end
        else
            if x  <= 0.0
                return cons_l
            else
                return cons_r
            end
        end
    end
    init_state!(app, mesh, state_prognostic_0, shock_tube_func)
    set_init_state!(solver, state_prognostic_0)
    
    visual(mesh, state_prognostic_0[:,1,:], "Sod_init_"*direction*".png")
    
    
    Q = solve!(solver)
    
    
    visual(mesh, Q[:,1,:], "Sod_end_"*direction*".png")



    state_primitive = solver.state_primitive
    prog_to_prim!(app, Q, solver.state_auxiliary_vol_l, state_primitive)
    if direction == "vertical"
        coord = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[1, :]
        vel  = reshape(state_primitive[:, 3 ,:], (Nl * Nx, Nz))[1, :]
    else
        coord = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, 1] .+ Lx/2.0
        vel  = reshape(state_primitive[:, 2 ,:], (Nl * Nx, Nz))[:, 1]
    end
    return coord, vel
    
    
end


vertical_method = "WENO3"
coor_WENO3, qoi_WENO3 = shock_tube("vertical", vertical_method)
vertical_method = "WENO5"
coor_WENO5, qoi_WENO5 = shock_tube("vertical", vertical_method)
vertical_method = "FV"
coor_FV, qoi_FV = shock_tube("vertical", vertical_method)
coor_DG, qoi_DG = shock_tube("horizontal", vertical_method, 2)


PyPlot.plot(coor_FV, qoi_FV, "-o", fillstyle = "none", label = "FV")
PyPlot.plot(coor_WENO3, qoi_WENO3, "-o", fillstyle = "none", label = "WENO3")
PyPlot.plot(coor_WENO5, qoi_WENO5, "-o", fillstyle = "none", label = "WENO5")
PyPlot.plot(coor_DG, qoi_DG, "-o", fillstyle = "none", label = "DG (p=2 overintegration)")

PyPlot.legend()
PyPlot.savefig("Shock-Tube-Vel.pdf")
