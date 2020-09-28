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
app = Adv("periodic", "periodic", 1.0, 1.0)


@show mesh.Δs_min[1, :, :]
@show mesh.Δs_min[2, :, :]


# initial condition
num_state_prognostic, nelem = app.num_state_prognostic, Nx*Nz
state_prognostic_0 = ones(Nl, num_state_prognostic, nelem)

cfl0 = 1.0/Np
dt = compute_cfl_dt(app, mesh, state_prognostic_0, cfl0)
@info "dt0 is ", dt
params = Dict("Time_Integrator" => "RK2", "cfl" => -1.0, "dt0" => dt, "t_end" => 10.0)


solver = Solver(app, mesh, state_prognostic_0, params)


Q = solve!(solver)



