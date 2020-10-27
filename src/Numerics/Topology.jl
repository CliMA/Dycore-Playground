import PyPlot
"""
LES configuration
Nl: number of Gauss-Legendre-Lobatto points in each element
Nx: horizontal element number 
Nz: vertical element number

The unwraped domain is (x, z) ∈ [-Lx/2, Lx/2]×[0, Ly]
"""
function topology_les(Nl::Int64, Nx::Int64, Nz::Int64, Lx::Float64, Lz::Float64, 
    xx = Array(LinRange(-Lx/2.0, Lx/2.0, Nx+1))::Array{Float64, 1},  zz=Array(LinRange(0, Lz, Nz+1))::Array{Float64, 1})
    
    dim = 2
    topology = zeros(Float64, dim, (Nl - 1)*Nx + 1, Nz + 1)
    

    ξl, ωl = lglpoints(Nl - 1)
    
    zp = zz
    for i = 1:size(topology, 2)
        topology[2, i, :] .= zp
    end
    
    
    xp = zeros(Float64, (Nl - 1)*Nx + 1)
    for ix = 1:Nx
        # x positions
        Δx = xx[ix + 1] - xx[ix]
        xm = (xx[ix] + xx[ix+1])/2.0
        xp[(Nl - 1)*(ix-1) + 1 : (Nl - 1)*ix + 1] .= xm .+  ξl * Δx/2.0
    end
    
    for i = 1:size(topology, 3)
        topology[1, :, i] .= xp
    end
    
    return topology
end


function mountain_wrap_les!(Nl::Int64, Nx::Int64, Nz::Int64, Lx::Float64, Lz::Float64, topology::Array{Float64, 3})

    
    dim = 2
    @assert(size(topology) == (dim, (Nl - 1)*Nx + 1, Nz + 1))

    # zb
    zb = zeros(Float64, (Nl - 1)*Nx + 1)
    xb = topology[1, :, 1]

    h, a, λ = 2.5e2 , 5e3 , 4e3
    zb = h*exp.(-xb.^2/a^2) .* cos.(π*xb/λ).^2

    # h, a = 1 , 1e3 
    # zb = h*a*a/(xb.^2 .+ a^2)

    for ix = 1:(Nl - 1)*Nx + 1
        topology[2, ix, :] = zb[ix] .+ topology[2, ix, :]/ Lz *(Lz - zb[ix])
    end
    
end


function mountain_wrap_les!(Nl::Int64, Nx::Int64, Nz::Int64, Lx::Float64, Lz::Float64, topology::Array{Float64, 3}, f_topology::Function)

    
    dim = 2
    @assert(size(topology) == (dim, (Nl - 1)*Nx + 1, Nz + 1))

    # zb
    xb = topology[1, :, 1]

    zb = f_topology(xb)

    # h, a = 1 , 1e3 
    # zb = h*a*a/(xb.^2 .+ a^2)

    for ix = 1:(Nl - 1)*Nx + 1
        topology[2, ix, :] = zb[ix] .+ topology[2, ix, :]/ Lz *(Lz - zb[ix])
    end
    
end

"""
GCM~(Arch) configuration
Nl: number of Gauss-Legendre-Lobatto points in each element
Nx: horizontal element number 
Nz: vertical element number

The unwraped domain is (r, θ) ∈ [r, R]×[0, 2π]
"""
function topology_gcm(Nl::Int64, Nx::Int64, Nz::Int64, r::Float64, R::Float64, 
         rr = Array(LinRange(r, R,  Nz+1))::Array{Float64, 1},
         θθ = Array(LinRange(0, -2π, Nx+1))::Array{Float64, 1})
    # The unwraped domain is [-Lx/2, Lx/2]×[0, Ly], the mesh is uniform in the horizontal direction
    dim = 2
    topology = zeros(Float64, dim, (Nl - 1)*Nx + 1, Nz + 1)
    top_ghost_cell = zeros(Float64, dim, Nx)

    ξl, ωl = lglpoints(Nl - 1)
    
    

    rp = rr
    θp = zeros(Float64, (Nl - 1)*Nx + 1)

    for ix = 1:Nx
        # x positions
        Δθ = θθ[ix + 1] - θθ[ix]
        θm = (θθ[ix] + θθ[ix+1])/2.0
        θp[(Nl - 1)*(ix-1) + 1 : (Nl - 1)*ix + 1] .= θm .+  ξl * Δθ/2.0
    end

    for ix = 1:size(topology, 2)
        for iz = 1:size(topology, 3)
            topology[:, ix, iz] .= rp[iz]*cos(θp[ix]), rp[iz]*sin(θp[ix])
        end
    end
    
    return topology
end


function vis_topology(Nx::Int64, Nz::Int64, topology::Array{Float64, 3}, save_file_name::String="None")
    Nl = Int64((size(topology, 2) - 1) /Nx) + 1

    # All Gauss-Legendre-Lobatto points
    x, z = topology[1,:,:], topology[2,:,:]

    # Grid points
    xg, zg = topology[1,1:Nl-1:end,:], topology[2,1:Nl-1:end,:]
    
    
    # Plot the mesh
    PyPlot.plot(xg, zg, color="b", linewidth=1) # use plot, not scatter
    PyPlot.plot(xg', zg', color="b", linewidth=1)

    

    # Plot the Gauss-Legendre-Lobatto points (high order) mesh
    PyPlot.plot(x, z, color="red", linewidth=1, linestyle="--") # use plot, not scatter
    PyPlot.plot(x', z', color="red", linewidth=1, linestyle="--")

    

    PyPlot.axis("equal")
    if save_file_name != "None"

        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
end
