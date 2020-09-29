include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")



Np = 3
Nl = Np+1
Nq = ceil(Int64, (3*Np + 1)/2)

Nx, Nz = 8,4
Lx, Lz = 1.0, 1.0

topology_type = "AtmoLES"
topology_size = [Lx; Lz]
topology = Topology(Nl, Nx, Nz, Lx, Lz)



mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)

gravity = false
app = DryEuler("periodic", "periodic", gravity)



params = Dict("Time_Integrator" => "RK2", "cfl" => -1.0, "dt0" => -1.0, "t_end" => 10.0)
solver = Solver(app, mesh, state_prognostic_0, params)



# update constant dt based on cfl
cfl0 = 1.0/Np
dt = compute_cfl_dt(app, mesh, state_prognostic_0, solver.state_auxiliary_vol_q, cfl0)
solver.dt0 = dt
@info "dt0 is ", dt

# update initial condition 
state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
prim_l = [1.0;   0.0; 0.0; 1.0]
prim_r = [0.125; 0.0; 0.0; 0.1]
cons_l = prim_to_prog(app, prim_l, zeros(Float64, app.num_state_auxiliary))
cons_r = prim_to_prog(app, prim_r, zeros(Float64, app.num_state_auxiliary))

function shock_tube_func(x::Float64, z::Float64)
    if z  <= Lz/2.0
        return cons_l
    else
        return cons_r
    end
end
init_state!(app, mesh, state_prognostic_0, shock_tube_func)
solver.state_prognostic .= state_prognostic_0

Q = solve!(solver)


