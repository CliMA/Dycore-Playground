include("../src/Numerics/Mesh.jl")
include("../src/Apps/Application.jl")
include("../src/Numerics/Solver.jl")



Np = 3
Nl = Np+1
Nq = ceil(Int64, (3*Np + 1)/2)

Nx, Nz = 2,2
Lx, Lz = 1.0, 1.0

topology_type = "AtmoLES"
topology_size = [Lx; Lz]
topology = Topology(Nl, Nx, Nz, Lx, Lz)



mesh = Mesh(Nx, Nz, Nl, Nq, topology_type, topology_size, topology)
app = Adv("periodic", "periodic", 1.0, 1.0)

dt = 0.1
params = Dict("Time_Integrator" => "RK2", "cfl_freqency" => -1,  "cfl" => 1/Np, "dt0" => dt, "t_end" => 10.0)
solver = Solver(app, mesh, params)

# initial condition
num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)
set_init_state!(solver, state_prognostic_0)

# vol_l_geo = mesh.vol_l_geo
# @show vol_l_geo
# error("stop")






Q = solve!(solver)



