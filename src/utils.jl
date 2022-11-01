function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nsrc"
            help = "Number of sources"
            arg_type = Int
            default = 16
        "--length"
            help = "Max length to cut or pad for velocity models"
            arg_type = Int
            default = 20
        "--ncont"
            help = "Number of background models for each compass slice"
            arg_type = Int
            default = 200
    end
    return parse_args(s)
end

function gaussian_background(m::Matrix{Float32}, width::Union{Number, Tuple{Number, Number}}; idx_wb=idx_wb)
    v =  1f0./sqrt.(m);
    v0 = deepcopy(v);
    v0[:,idx_wb+1:end] = imfilter(v[:,idx_wb+1:end], Kernel.gaussian(width))
    m0 = (1f0 ./ v0).^2
    return m0
end

function gen_m0_vary(m; idx_wb=idx_wb, d=d, lengthmax=15)
    n = size(m)
    X = convert(Array{Float32},reshape(range(0f0,stop=(n[1]-1)*d[1],length=n[1]),:,1))
    Cova = gaussian_kernel(X,X',theta0=5,delta=250,cons=1f-5)
    cutlength = rand(MvNormal(zeros(Float32,n[1]),Cova))
    cutlength = abs.(Int.(round.(cutlength/norm(cutlength,Inf) * lengthmax)))
    start_cut = rand(idx_wb+1:n[2]-maximum(cutlength))
    if rand()>=0.5      # REMOVE
        println("remove")
        keep_idx = [deleteat!(collect(idx_wb+1:n[2]), start_cut-idx_wb:start_cut+cutlength[i]-1-idx_wb) for i = 1:n[1]]
        m0 = vcat([vcat(m[i,1:idx_wb],imresize(m[i,keep_idx[i]], n[2]-idx_wb)) for i = 1:n[1]]'...)
    else                # PAD
        println("pad")
        add_idx = [vcat(collect(idx_wb+1:start_cut), start_cut+1:start_cut+cutlength[i], start_cut+1:n[2]) for i = 1:n[1]]
        m0 = vcat([vcat(m[i,1:idx_wb],imresize(m[i,add_idx[i]], n[2]-idx_wb)) for i = 1:n[1]]'...)
    end
    return gaussian_background(m0, 20)
end

function gaussian_kernel(xa,xy;theta0=1,delta=1,cons=1f-5)
    return theta0*exp.(-(xa.-xy).^2f0*1f0/delta^2f0)+theta0*cons*I
end

function mute_turning(d_obs::judiVector{Float32, Matrix{Float32}}, q::judiVector{Float32, Matrix{Float32}}; t0::Number=1f-1, v_water::Number=1480f0)
    d_out = deepcopy(d_obs)
    for i = 1:d_out.nsrc
        xsrc = q.geometry.xloc[i][1]
        zsrc = q.geometry.zloc[i][1]
        for j = 1:size(d_out.data[i],2)
            xrec = d_out.geometry.xloc[i][j]
            zrec = d_out.geometry.zloc[i][j]
            direct_time = sqrt((xsrc-xrec)^2f0+(zsrc-zrec)^2f0)/v_water + t0 # in s
            mute_end = min(Int(floor(direct_time*1f3/d_out.geometry.dt[i])),size(d_out.data[i],1))
            d_out.data[i][1:mute_end,j] .= 0
        end
    end
    return d_out
end
