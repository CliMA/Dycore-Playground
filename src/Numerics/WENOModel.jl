include("Mesh.jl")
include("../Apps/Application.jl")

"""
Observations on the fifth-order WENO method with non-uniform meshes

size(Δh) = 5
size(u) = (num_state_prognostic, 5)
construct left/right face states of cell[3]

h1     h2     h3     h4      h5
|--i-2--|--i-1--|--i--|--i+1--|--i+2--|
without hat : i - 1/2
with hat    : i + 1/2  


r = 0, 1, 2 cells to the left, => P_r^{i}
I_{i-r}, I_{i-r+1}, I_{i-r+2}

P_r(x) =  ∑_{j=0}^{2} C_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
C_{rj}(x) = B_{rj}(x) h_{3-r+j}                (i = 3)


C''_{rj}(x) = B''_{rj}(x) h_{3-r+j}                (i = 3)
P''_r(x)    =  ∑_{j=0}^{2} C''_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
=  ∑_{j=0}^{2} B''_{rj}(x) h_{3-r+j}  u_{i - r + j}  (use these 3 cell averaged values)
"""
function weno5_recon(h::Array{Float64, 1}, u::Array{Float64, 2})
    num_state_prognostic = size(u, 1)
    h1, h2, h3, h4, h5 = h
    
    b̂ = zeros(Float64, 3, 3)
    b̂[3, 3] = 1/(h1+h2+h3) + 1/(h2+h3) + 1/h3
    b̂[3, 2] = b̂[3,3] - (h1+h2+h3)*(h2+h3)/((h1+h2)*h2*h3)
    b̂[3, 1] = b̂[3, 2] + (h1+h2+h3)*h3/(h1*h2*(h2+h3))
    b̂[2, 3] = (h2+h3)*h3/((h2+h3+h4)*(h3+h4)*h4)
    b̂[2, 2] = b̂[2, 3] + 1/(h2+h3) + 1/h3 - 1/h4
    b̂[2, 1] = b̂[2, 2] - ((h2+h3)*h4)/(h2*h3*(h3+h4))
    b̂[1, 3] = -(h3*h4)/((h3+h4+h5)*(h4+h5)*h5)
    b̂[1, 2] = b̂[1, 3] + h3*(h4+h5)/((h3+h4)*h4*h5) 
    b̂[1, 1] = b̂[1, 2] + 1/h3 - 1/h4 - 1/(h4+h5)
    
    b = zeros(Float64, 3, 3)
    b[3, 3] = 1/(h5+h4+h3) + 1/(h4+h3) + 1/h3
    b[3, 2] = b[3,3] - (h5+h4+h3)*(h4+h3)/((h5+h4)*h4*h3)
    b[3, 1] = b[3, 2] + (h5+h4+h3)*h3/(h5*h4*(h4+h3))
    
    b[2, 3] = (h4+h3)*h3/((h4+h3+h2)*(h3+h2)*h2)
    b[2, 2] = b[2, 3] + 1/(h4+h3) + 1/h3 - 1/h2
    b[2, 1] = b[2, 2] - ((h4+h3)*h2)/(h4*h3*(h3+h2))
    b[1, 3] = -(h3*h2)/((h3+h2+h1)*(h2+h1)*h1)
    b[1, 2] = b[1, 3] + h3*(h2+h1)/((h3+h2)*h2*h1) 
    b[1, 1] = b[1, 2] + 1/h3 - 1/h2 - 1/(h2+h1)
    
    
    # at i - 1/2, i + 1/2
    P =  zeros(Float64, num_state_prognostic, 2, 3)
    for r = 0:2
        for j = 0:2
            P[:, 1, 3-r] += b[r+1, j+1] * h[3+r-j] * u[:, 3+r-j]
            P[:, 2, r+1] += b̂[r+1, j+1] * h[3-r+j] * u[:, 3-r+j]
        end
    end
    
    
    # build the second derivative part in smoothness measure
    d2B = zeros(Float64, 3, 3)
    for r = 0:2
        d2B[r+1, 3] = 6.0/((h[3-r] + h[4-r] + h[5-r])*(h[4-r] + h[5-r])*h[5-r])
        d2B[r+1, 2] = d2B[r+1, 3]  - 6.0/((h[3-r] + h[4-r]) * h[4-r] * h[5-r])
        d2B[r+1, 1] = d2B[r+1, 2]  + 6.0/(h[3-r] * h[4-r] * (h[4-r] + h[5-r]))
    end
    
    d2P = zeros(Float64, num_state_prognostic, 3)
    for r = 0:2
        for j = 0:2
            d2P[:, r+1] += d2B[r+1, j+1]*h[3-r+j]*u[:, 3-r+j]
        end
    end
    
    IS2 = h3^4 * d2P.^2
    
    # build the first derivative part in smoothness measure
    
    d1B = zeros(Float64, 3, 3, 3) # xi-1/2 xi, xi+1/2; r, j
    d1B[1, 3, 3] = 2*(h1 + 2*h2)/((h1+h2+h3)*(h2+h3)*h3)
    d1B[1, 3, 2] = d1B[1, 3, 3] - 2*(h1 + 2*h2 - h3)/((h1+h2)*h2*h3)
    d1B[1, 3, 1] = d1B[1, 3, 2] + 2*(h1 + h2 - h3)/(h1*h2*(h2+h3))
    
    d1B[1, 2, 3] = 2*(h2 - h3)/((h2+h3+h4)*(h3+h4)*h4)
    d1B[1, 2, 2] = d1B[1, 2, 3] - 2*(h2 - h3 - h4)/((h2+h3)*h3*h4)
    d1B[1, 2, 1] = d1B[1, 2, 2] + 2*(h2 - 2*h3 - h4)/(h2*h3*(h3+h4))
    
    # bug in the paper
    d1B[1, 1, 3] = -2*(2*h3 + h4)/((h3+h4+h5)*(h4+h5)*h5)
    d1B[1, 1, 2] = d1B[1, 1, 3] + 2*(2*h3 + h4 + h5)/((h3+h4)*h4*h5)
    d1B[1, 1, 1] = d1B[1, 1, 2] - 2*(2*h3 + 2*h4 + h5)/(h3*h4*(h4+h5))
    
    for r = 0:2
        for j = 0:2
            d1B[2, r+1, j+1] = d1B[1, r+1, j+1] + 0.5*h3*d2B[r+1, j+1]
            d1B[3, r+1, j+1] = d1B[1, r+1, j+1] + h3*d2B[r+1, j+1]
        end
    end
    d1P = zeros(Float64, num_state_prognostic, 3, 3)   # xi-1/2 xi, xi+1/2; r 
    for i = 1:3
        for r = 0:2
            for j = 0:2
                d1P[:, i, r+1] += d1B[i, r+1, j+1] * h[3-r+j] * u[:, 3-r+j]
            end
        end
    end
    
    IS1 = h3^2 * (d1P[:, 1, :].^2 + 4*d1P[:, 2, :].^2 + d1P[:, 3, :].^2  )/6.0
    
    
    IS = IS1 + IS2
    
    d = zeros(Float64, 3)
    d[3] = (h3+h4)*(h3+h4+h5)/((h1+h2+h3+h4)*(h1+h2+h3+h4+h5))
    d[2] = (h1+h2)*(h3+h4+h5)*(h1+2*h2+2*h3+2*h4+h5)/((h1+h2+h3+h4)*(h2+h3+h4+h5)*(h1+h2+h3+h4+h5))
    d[1] = h2*(h1+h2)/((h2+h3+h4+h5)*(h1+h2+h3+h4+h5))
    
    d̂ = zeros(Float64, 3)
    d̂[3] = h4*(h4+h5)/((h1+h2+h3+h4)*(h1+h2+h3+h4+h5))
    d̂[2] = (h1+h2+h3)*(h4+h5)*(h1+2*h2+2*h3+2*h4+h5)/((h1+h2+h3+h4)*(h2+h3+h4+h5)*(h1+h2+h3+h4+h5))
    d̂[1] = (h2+h3)*(h1+h2+h3)/((h2+h3+h4+h5)*(h1+h2+h3+h4+h5))
    
    ϵ = 1.0e-6
    α = d'./(ϵ .+ IS).^2
    α̂ = d̂'./(ϵ .+ IS).^2
    
    w = α ./ sum(α, dims=2)
    ŵ = α̂ ./ sum(α̂, dims=2)
    
    
    # at  i - 1/2,  i + 1/2
    u⁻ = zeros(Float64, num_state_prognostic)
    u⁺ = zeros(Float64, num_state_prognostic)
    for i = 1:num_state_prognostic
        u⁻[i] += w[i, :]' * P[i, 1, :]
        u⁺[i] += ŵ[i, :]' * P[i, 2, :]
    end
    
    return u⁻, u⁺
    
    
end

"""
h1     h2     h3    
|--i-1--|--i--|--i+1--|
without hat : i - 1/2
with hat    : i + 1/2  
"""
function weno3_recon(h::Array{Float64, 1}, u::Array{Float64, 2})
    num_state_prognostic = size(u, 1)
    h1, h2, h3 = h
    
    
    β , γ = h1/h2, h3/h2
    C⁺_l, C⁺_r = γ/(1 + β + γ)    ,     (1 + β)/(1 + β + γ)
    C⁻_l, C⁻_r = (1+β)/(1 + β + γ),      γ/(1 + β + γ)
    
    
    dP =  zeros(Float64, num_state_prognostic, 2)
    dP[:, 1] = 2*(u[:, 2] - u[:,1])/(h1 + h2)
    dP[:, 2] = 2*(u[:, 3] - u[:,2])/(h2 + h3)
    
    
    # at i - 1/2, i + 1/2, r = 0, 1
    P =  zeros(Float64, num_state_prognostic, 2, 2)
    for r = 0:1
        P[:, 1, r+1] = u[:, 2]  - dP[:, r+1]* h2/2.0
        P[:, 2, r+1] = u[:, 2]  + dP[:, r+1]* h2/2.0
    end
    
    # IS = int h2 *P'^2 dx, P  is a linear function
    IS = h2^2 * dP.^2 
    
    # todo debug
    # IS .= 1.0
    
    d = [(1+γ)/(1 + β + γ)  ;     β/(1 + β + γ)]
    d̂ = [γ/(1 + β + γ)    ;     (1 + β)/(1 + β + γ)]
    
    ϵ = 1.0e-6
    α = d'./(ϵ .+ IS).^2
    α̂ = d̂'./(ϵ .+ IS).^2
    
    w = α ./ sum(α, dims=2)
    ŵ = α̂ ./ sum(α̂, dims=2)
    
    # at  i - 1/2,  i + 1/2
    u⁻ = zeros(Float64, num_state_prognostic)
    u⁺ = zeros(Float64, num_state_prognostic)
    for i = 1:num_state_prognostic
        u⁻[i] += w[i, :]' * P[i, 1, :]
        u⁺[i] += ŵ[i, :]' * P[i, 2, :]
    end
    
    return u⁻, u⁺
end

function reconstruction_1d_weno3(app::Application, state_primitive_col, Δzc_col, 
    bc_bottom_type::String, bc_bottom_data::Union{Array{Float64, 1}, Nothing}, bc_bottom_n::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String, bc_top_data::Union{Array{Float64, 1}, Nothing}, bc_top_n::Union{Array{Float64, 1}, Nothing},
    state_primitive_face⁻::Array{Float64, 2}, state_primitive_face⁺::Array{Float64, 2})
    
    num_state_prognostic, Nz = size(state_primitive_col)
    
    
    ##########################################################################################################
    # compute face states by looping cells
    num_left_stencil = 1
    state_primitive_weno, Δz_weno = zeros(num_state_prognostic, 2num_left_stencil+1), zeros(2num_left_stencil+1)
    for iz = 1:Nz
        for is = 1: 2num_left_stencil+1
            state_primitive_weno[:, is] = state_primitive_col[:, mod1(iz - num_left_stencil + is - 1, Nz)]
            Δz_weno[is] =  Δzc_col[mod1(iz - num_left_stencil + is - 1, Nz)]
        end
        (state_primitive_face⁺[:, iz], state_primitive_face⁻[:, iz+1])   = weno3_recon(Δz_weno, state_primitive_weno)
    end
    if bc_bottom_type == "periodic" && bc_top_type == "periodic"
        state_primitive_face⁻[:, 1] .=  state_primitive_face⁻[:, Nz+1]
        state_primitive_face⁺[:, Nz+1] .= state_primitive_face⁺[:, 1] 
        return ;
    end
    
    # update other type of boundary conditions
    if bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
        Δz1, Δz2 = Δzc_col[1], Δzc_col[2]
        # one-side extrapolation
        state_primitive_face⁺[:, 1] = (2*Δz1+Δz2)/(Δz1+Δz2)*state_primitive_col[:, 1] - Δz1/(Δz1+Δz2)*state_primitive_col[:, 2]
        state_primitive_face⁺[:, 1] = bc_impose(app, state_primitive_face⁺[:, 1], bc_bottom_type, bc_bottom_n)
        # interpolation
        
        Δstate⁺ = state_primitive_col[:, 2] - state_primitive_col[:, 1]
        Δstate⁻ = state_primitive_col[:, 1] - state_primitive_face⁺[:, 1]
        Δz⁺ = (Δz1 + Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        state_primitive_face⁻[:, 2] = state_primitive_col[:, 1] + ∂state * Δz1/2.0
        
    else
        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
    end
    
    # populate top ghost state
    if bc_top_type == "outlet"
        Δz1, Δz2 = Δzc_col[Nz], Δzc_col[Nz-1]
        # outlet state
        state_primitive_face⁺[:, Nz + 1] = bc_top_data
        
        Δstate⁺ = state_primitive_col[:, Nz] - state_primitive_col[:, Nz-1]
        Δstate⁻ = state_primitive_face⁺[:, Nz + 1] - state_primitive_col[:, Nz] 
        Δz⁺ = (Δz1+Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        
        # one-side extrapolation
        state_primitive_face⁻[:, Nz + 1] = state_primitive_col[:, Nz] + ∂state * Δz1/2.0
        state_primitive_face⁺[:, Nz] = state_primitive_col[:, Nz] - ∂state * Δz1/2.0
    else
        error("bc_top_type = ", bc_top_type, " has not implemented")   
    end   
end




function reconstruction_1d_weno5(app::Application, state_primitive_col, Δzc_col, 
    bc_bottom_type::String, bc_bottom_data::Union{Array{Float64, 1}, Nothing}, bc_bottom_n::Union{Array{Float64, 1}, Nothing},
    bc_top_type::String, bc_top_data::Union{Array{Float64, 1}, Nothing}, bc_top_n::Union{Array{Float64, 1}, Nothing},
    state_primitive_face⁻::Array{Float64, 2}, state_primitive_face⁺::Array{Float64, 2})
    
    num_state_prognostic, Nz = size(state_primitive_col)
    
    
    ##########################################################################################################
    # compute face states by looping cells
    num_left_stencil = 2
    state_primitive_weno, Δz_weno = zeros(num_state_prognostic, 2num_left_stencil+1), zeros(2num_left_stencil+1)
    for iz = 1:Nz
        for is = 1: 2num_left_stencil+1
            state_primitive_weno[:, is] = state_primitive_col[:, mod1(iz - num_left_stencil + is - 1, Nz)]
            Δz_weno[is] =  Δzc_col[mod1(iz - num_left_stencil + is - 1, Nz)]
        end
        (state_primitive_face⁺[:, iz], state_primitive_face⁻[:, iz+1]) = weno5_recon(Δz_weno, state_primitive_weno)
    end
    if bc_bottom_type == "periodic" && bc_top_type == "periodic"
        state_primitive_face⁻[:, 1] .=  state_primitive_face⁻[:, Nz+1]
        state_primitive_face⁺[:, Nz+1] .= state_primitive_face⁺[:, 1] 
        return ;
    end

    # reduce to weno3 on near the bc
    num_left_stencil = 1
    state_primitive_weno, Δz_weno = zeros(num_state_prognostic, 2num_left_stencil+1), zeros(2num_left_stencil+1)
    for iz = [2, Nz-1]
        for is = 1: 2num_left_stencil+1
            state_primitive_weno[:, is] = state_primitive_col[:, (iz - num_left_stencil + is - 1)]
            Δz_weno[is] =  Δzc_col[(iz - num_left_stencil + is - 1)]
        end
        (state_primitive_face⁺[:, iz], state_primitive_face⁻[:, iz+1]) = weno3_recon(Δz_weno, state_primitive_weno)
    end

    # reduce to fv at bc
    # update other type of boundary conditions
    if bc_bottom_type == "no-slip" || bc_bottom_type == "no-penetration"
        Δz1, Δz2 = Δzc_col[1], Δzc_col[2]
        # one-side extrapolation
        state_primitive_face⁺[:, 1] = (2*Δz1+Δz2)/(Δz1+Δz2)*state_primitive_col[:, 1] - Δz1/(Δz1+Δz2)*state_primitive_col[:, 2]
        state_primitive_face⁺[:, 1] = bc_impose(app, state_primitive_face⁺[:, 1], bc_bottom_type, bc_bottom_n)
        # interpolation
        
        Δstate⁺ = state_primitive_col[:, 2] - state_primitive_col[:, 1]
        Δstate⁻ = state_primitive_col[:, 1] - state_primitive_face⁺[:, 1]
        Δz⁺ = (Δz1 + Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        state_primitive_face⁻[:, 2] = state_primitive_col[:, 1] + ∂state * Δz1/2.0
        
    else
        error("bc_bottom_type = ", bc_bottom_type, " has not implemented")   
    end
    
    # populate top ghost state
    if bc_top_type == "outlet"
        Δz1, Δz2 = Δzc_col[Nz], Δzc_col[Nz-1]
        # outlet state
        state_primitive_face⁺[:, Nz + 1] = bc_top_data
        
        Δstate⁺ = state_primitive_col[:, Nz] - state_primitive_col[:, Nz-1]
        Δstate⁻ = state_primitive_face⁺[:, Nz + 1] - state_primitive_col[:, Nz] 
        Δz⁺ = (Δz1+Δz2)/2.0
        Δz⁻ = Δz1/2.0
        ∂state = limiter(Δstate⁺/Δz⁺, Δstate⁻/Δz⁻)
        
        # one-side extrapolation
        state_primitive_face⁻[:, Nz + 1] = state_primitive_col[:, Nz] + ∂state * Δz1/2.0
        state_primitive_face⁺[:, Nz] = state_primitive_col[:, Nz] - ∂state * Δz1/2.0
    else
        error("bc_top_type = ", bc_top_type, " has not implemented")   
    end   
end





function WENO_Test()
    
    function lin_func(ξ)
        return [(2*ξ .+ 1.0)';] , [(ξ.^2 .+ ξ)';] 
    end
    
    function quad_func(ξ)
        return [(3.0*ξ.^2 .+ 1.0)';(2*ξ .+ 1.0)';] , [(ξ.^3 .+ ξ)';(ξ.^2 .+ ξ)';] 
    end
    
    function third_func(ξ)
        return [(4.0*ξ.^3 .+ 1.0)';] , [(ξ.^4 .+ ξ)';] 
    end
    
    function fourth_func(ξ)
        return [(5.0*ξ.^4 .+ 1.0)';(3.0*ξ.^2 .+ 1.0)'] , [(ξ.^5 .+ ξ)';(ξ.^3 .+ ξ)'] 
    end
    
    ## weno3 test pass, set IS .= 1.0, leads to p2 recovery 
    grid = [0.0; 1.0; 3.0; 6.0]*0.1
    h =  grid[2:end] - grid[1:end-1]
    grid_c = (grid[2:end] + grid[1:end-1])/2.0
    func = quad_func
    uc, uc_I = func(grid_c)
    uf, uf_I = func(grid)
    u = similar(uc)
    for i =1:length(h)
        u[:, i] = ( uf_I[:, i+1] - uf_I[: , i] )/h[i]
    end
    u⁻, u⁺ = weno3_recon(h, u)
    @info "uf_ref : ", uf[:, 2], uf[:, 3]
    @info "u⁻, u⁺ : ", u⁻, u⁺ 
    
    ## weno5 test, set IS .= 1.0, leads to p4 recovery 
    grid = [0.0; 1.0; 3.0; 6.0; 7.0; 9.0]*0.1
    h =  grid[2:end] - grid[1:end-1]
    grid_c = (grid[2:end] + grid[1:end-1])/2.0
    func = fourth_func
    uc, uc_I = func(grid_c)
    uf, uf_I = func(grid)
    u = similar(uc)
    for i =1:length(h)
        u[:, i] = ( uf_I[:, i+1] - uf_I[: , i] )/h[i]
    end
    u⁻, u⁺ = weno5_recon(h, u)
    @info "uf_ref : ", uf[:, 3], uf[:, 4]
    @info "u⁻, u⁺ : ", u⁻, u⁺ 
end
