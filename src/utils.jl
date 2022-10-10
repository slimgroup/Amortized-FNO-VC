function λtoΣ(λ::Vector{Float32})
    A = randn(Float32,2,2)
    Q, _ = qr(A)
    return Q' * diagm(λ) * Q
end
function gaussianfunction(x, y, Σ::Matrix{Float32})
    return .5f0/pi/√(abs(det(Σ))) * exp(-0.5f0 * dot([x, y], Σ\[x;y]))
end
function gaussiankernel(Σ::Matrix{Float32})
    width = (40,40)
    A = zeros(Float32, width)
    
    for i = 1:width[1]
        for j = 1:width[2]
            A[i,j] = gaussianfunction(i-(width[1]+1)/2f0, j-(width[2]+1)/2f0, Σ)
        end
    end
    return A/sum(A)
end
function gaussian_background(m::Matrix{Float32}, width::Union{Number, Tuple{Number, Number}}; idx_wb=idx_wb)
    v =  1f0./sqrt.(m);
    v0 = deepcopy(v);
    v0[:,idx_wb+1:end] = imfilter(v[:,idx_wb+1:end], Kernel.gaussian(width))
    m0 = (1f0 ./ v0).^2
    return m0
end
function gaussian_background(m::Matrix{Float32}, Σ::Matrix{Float32}; idx_wb=idx_wb)
    v =  1f0./sqrt.(m);
    v0 = deepcopy(v);
    v0[:,idx_wb+1:end] = imfilter(v[:,idx_wb+1:end], gaussiankernel(Σ))
    m0 = (1f0 ./ v0).^2
    return m0
end
function gen_m0(m; cut_rows=100, idx_wb=idx_wb)
    keep_rows = deleteat!(collect(idx_wb+1:size(m,2)), sort(randperm(size(m,2)-idx_wb)[1:cut_rows]))
    m0 = hcat(m[:,1:idx_wb],imresize(m[:,keep_rows], (size(m,1), size(m,2)-idx_wb)))
    return gaussian_background(m0, 20)
end
function gen_m0_vary(m; idx_wb=idx_wb, d=d)
    n = size(m)
    X = convert(Array{Float32},reshape(range(0f0,stop=(n[1]-1)*d[1],length=n[1]),:,1))
    Cova = gaussian_kernel(X,X',theta0=5,delta=250,cons=1f-5)
    cutlength = rand(MvNormal(zeros(Float32,n[1]),Cova))
    cutlength = Int.(round.(cutlength/norm(cutlength) * 500 .+ 20))
    start_cut = rand(idx_wb+1:n[2]-maximum(cutlength))
    keep_idx = [deleteat!(collect(idx_wb+1:n[2]), start_cut-idx_wb:start_cut+cutlength[i]-1-idx_wb) for i = 1:n[1]]
    #λ = Float32((maximum(m)-minimum(m))/2/π * rand(Float32))
    m0 = vcat([vcat(m[i,1:idx_wb],imresize(m[i,keep_idx[i]], n[2]-idx_wb)) for i = 1:n[1]]'...)
    #m0 = vcat([vcat(m[i,1:idx_wb],v_m(skew.(m_v(imresize(m[i,keep_idx[i]], n[2]-idx_wb)), λ)))' for i = 1:n[1]]...)
    return gaussian_background(m0, 20)
end
function gaussian_kernel(xa,xy;theta0=1,delta=1,cons=1f-5)
    return theta0*exp.(-(xa.-xy).^2f0*1f0/delta^2f0)+theta0*cons*I
end
m_v(m::Array{Float32}) = 1f0./sqrt.(m)
v_m(v::Array{Float32}) = 1f0./v.^2f0
skew(x, λ::Float32, _min::Float32, _max::Float32) =  λ * Float32(sin(2 * π * (x-_min)/(_max-_min))) + x
skew(x, λ::Float32) = skew(x,λ,minimum(x),maximum(x))