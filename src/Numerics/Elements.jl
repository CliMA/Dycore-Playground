import GaussQuadrature
using LinearAlgebra

"""
Np is the polynomial order 
returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre-Lobatto quadrature rule of type `T`

"""
function lglpoints(Np::Int64)
    @assert Np ≥ 1
    GaussQuadrature.legendre(Float64, Np + 1, GaussQuadrature.both)
end

"""
Np is the polynomial order 

returns the points `r` and weights `w` associated with the `N+1`-point
Gauss-Legendre quadrature rule of type `T`
"""
function lgpoints(Np::Int64) 
    @assert Np ≥ 1
    GaussQuadrature.legendre(Float64, Np + 1, GaussQuadrature.neither)
end

"""
returns wb
wb(i) = 1/∏_{j≢i}(ξi - ξj)

Reference:
Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
SIAM Review 46 (2004), pp. 501-517.
<https://doi.org/10.1137/S0036144502417715>
"""
function baryweights(r::Array{Float64,1}) 
    Np = length(r)
    wb = ones(Float64, Np)
    
    for j in 1:Np
        for i in 1:Np
            if i != j
                wb[j] = wb[j] * (r[j] - r[i])
            end
        end
        wb[j] = Float64(1) / wb[j]
    end
    wb
end


"""
returns Nq × Nl = [ϕ_1 ; ϕ_2 ; ... ; ϕ_{N_l}]
wb(i) = 1/∏_{j≢i}(ξi - ξj)
l(ξ) =  ∏_{j}(ξ - ξj)     

ϕ_i(ξ) = ∏_{j≢i}(ξ - ξj)/∏_{j≢i}(ξi - ξj)
= l(ξ) * wb(i)/(ξ - ξi)

ϕ_i(ξ_iq) = ∏_{j≢i}(ξ_iq - ξj)*wb(i)


Reference:
Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
SIAM Review 46 (2004), pp. 501-517.
<https://doi.org/10.1137/S0036144502417715>
"""
function spectralinterpolate(rl::Array{Float64,1} , rq::Array{Float64,1}, wbl::Array{Float64,1})
    Nl = length(rl)
    Nq = length(rq)
    
    ϕl_q = zeros(Float64, Nq, Nl)
    
    for i = 1:Nl
        ϕl_q[:, i] .= wbl[i]
        for iq in 1:Nq
            for j = 1:Nl
                if j != i
                    ϕl_q[iq, i] *= (rq[iq] - rl[j])
                end
            end
        end
    end
    ϕl_q
end


"""
returns N_q × N_l = [Dϕ_1 ; Dϕ_2 ; ... ; Dϕ_{N_l}]
wb(i) = 1/∏_{j≢i}(ξi - ξj)
l(ξ) =  ∏_{j}(ξ - ξj)     

ϕ_i(ξ) = ∏_{j≢i}(ξ - ξj)/∏_{j≢i}(ξi - ξj)
= l(ξ) * wb(i)/(ξ - ξi)

Dϕ_i(ξ) = D ∏_{j≢i}(ξ - ξj)/∏_{j≢i}(ξi - ξj)
= ( ∑_{k≢i}  ∏_{j≢i,k}(ξ - ξj) )/∏_{j≢i}(ξi - ξj)
= ( ∑_{k≢i}  ∏_{j≢i,k}(ξ - ξj) )  * wb(i) 



Reference:
Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
SIAM Review 46 (2004), pp. 501-517.
<https://doi.org/10.1137/S0036144502417715>
"""
function spectralderivative(rl::Array{Float64,1} , rq::Array{Float64,1}, wbl::Array{Float64,1})
    Nl = length(rl)
    Nq = length(rq)
    
    Dl_q = zeros(Float64, Nq, Nl)
    
    for i = 1:Nl
        for iq in 1:Nq
            for k = 1:Nl
                if k != i
                    numer = 1.0
                    for j = 1:Nl
                        if j != i && j!=k
                            numer *= (rq[iq] - rl[j])
                        end
                    end
                    Dl_q[iq, i] +=  numer
                end
            end
            Dl_q[iq, i] *= wbl[i]
            
        end
    end
    
    return Dl_q
end





function Elements_test()
    
    Np = 3
    Nl = Np+1
    Nq = ceil(Int64, (3*Np + 1)/2)

    (ξl, ωl) = lglpoints(Nl - 1)
    (ξq, ωq) = lgpoints(Nq - 1)

    function func(ξ)
        return ξ.^Np + ξ .+ 1.0, Np*ξ.^(Np-1) .+ 1.0
    end

    fl, dfl = func(ξl)
    fq, dfq = func(ξq)

    wbl  = baryweights(ξl)
    Dl_l = spectralderivative(ξl, ξl, wbl)
    ϕl_l = spectralinterpolate(ξl, ξl, wbl)
    ϕl_q = spectralinterpolate(ξl, ξq, wbl)
    Dl_q = spectralderivative(ξl, ξq, wbl)

    fq_intp = ϕl_q * fl 
    fl_intp = ϕl_l * fl 
    dfl_intp = Dl_l * fl 
    dfq_intp = Dl_q * fl 

    

    @info "norm(fq - fq_intp) = ",  norm(fq - fq_intp)
    @info "norm(fl - fl_intp) = ",  norm(fl - fl_intp)
    @info "norm(dfl - dfl_intp) = ", norm(dfl - dfl_intp)
    @info "norm(dfq - dfq_intp) = ", norm(dfq - dfq_intp)
end

# Elements_test()


