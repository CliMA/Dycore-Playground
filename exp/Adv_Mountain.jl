include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")

"""
periodicity T = 1
"""
function f_topology(x::Array{Float64, 1})
    h0, a, λ, x0 = 3e3, 25e3, 8e3, 0.0
    nx = length(x)
    z = zeros(Float64, nx)

    for i = 1:nx
        if abs(x[i] - x0) <= a
            z[i] = h0 * cos(π*((x[i] - x0)/(2*a)))^2 * cos(π*((x[i] - x0)/(λ)))^2
        end
    end

    return z
end
function adv_mountain(vertical_method::String, t_end::Float64=3.0, Np::Int64=3, Nq::Int64=ceil(Int64, (3*Np + 1)/2))
    
    Nl = Np+1
    
    Nz = 60
    Nx = div(300, Np)
    Lx, Lz = 150e3, 30e3
    
    topology_type = "AtmoLES"
    topology_size = [Lx; Lz]
    topology = topology_les(Nl, Nx, Nz, Lx, Lz)
    mountain_wrap_les!(Nl, Nx, Nz, Lx, Lz, topology, f_topology)
    
    
    mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
    u, w, viscous, ν = 1.0, 0.0, false, NaN64
    app = AdvDiff("periodic", nothing,  # bottom
    "periodic", nothing,  # top
    "periodic", nothing,  # left
    "periodic", nothing,  # right
    u, w, viscous, ν)
    
    dt = 0.1
    params = Dict("time_integrator" => "RK4", "cfl_freqency" => -1,  "cfl" => 0.5, "dt0" => 10000.0, "t_end" => t_end, "vertical_method" => vertical_method)
    solver = Solver(app, mesh, params)
    
    # initial condition
    num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
    
    
    state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
    
    const_func = (x::Float64, z::Float64) -> 1.0
    function tracer_2d_func(x::Float64, z::Float64)
        rx, rz = 25.0e3, 3.0e3
        xc, zc = 25.0e3 - 75.0e3 , 9.0e3
        r = sqrt((x - xc)^2/rx^2 + (z - zc)^2/rz^2)

        if r <= 1
            return cos(π*r/2.0)^2
        else
            return 0.0
        end
    end
    

    

    init_state!(app, mesh, state_prognostic_0, tracer_2d_func)

    
    set_init_state!(solver, state_prognostic_0)
    
    # visualize the initial condition 
    visual(mesh, state_prognostic_0[:,1,:], "Adv_Mountain_init.png")
    
   
    Q = solve!(solver)
    
    # visualize the solution at t_end 
    visual(mesh, Q[:,1,:], "Adv_Mountain_end_"*vertical_method*".png")
    
    # coord_z = reshape(mesh.vol_l_geo[2,:,:], (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    # Q0_z  = reshape(state_prognostic_0, (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    # Q_z  = reshape(Q, (Nl * Nx, Nz))[div(Nl * Nx, 2), :]
    # coord_x = reshape(mesh.vol_l_geo[1,:,:], (Nl * Nx, Nz))[:, div(Nz, 2)] .+ Lx/2.0
    # Q_x  = reshape(Q, (Nl * Nx, Nz))[:, div(Nz, 2)]
    
    # PyPlot.figure()
    # PyPlot.plot(coord_z, Q0_z, "-o", fillstyle = "none", label = "Ref")
    # PyPlot.plot(coord_z, Q_z, "-o", fillstyle = "none", label = vertical_method)
    # PyPlot.plot(coord_x, Q_x, "-o", fillstyle = "none", label = "DG (p=3 overintegration)")
    # PyPlot.legend()
    # PyPlot.savefig("Adv_"*test_case*"_end_section"*vertical_method*".pdf")
    
    
    return 
    
    
end


adv_mountain("FV",    100000.0,   3)

# adv_mountain("WENO5",    100000.0,   3)
# adv_square("square", "WENO3", 3.0*period)
# adv_square("square", "WENO5", 3.0*period)


# adv_square("gaussian", "WENO3", 3.0*period)
# adv_square("gaussian", "WENO5", 3.0*period)
# adv_square("gaussian", "FV",    3.0*period)





