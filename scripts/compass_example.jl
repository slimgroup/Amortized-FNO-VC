# Author: Mathias Louboutin
# Date: June 2021
#

using DrWatson
@quickactivate "Amortized-FNO-VC"
using JUDI, LinearAlgebra, Images, PyPlot, DSP, ImageGather, Printf
using JLD2
using SlimPlotting
using Random
using Distributions
Random.seed!(2022)

include(srcdir("utils.jl"))
JLD2.@load "../data/BGCompass_tti_625m.jld2" n d o m;
extent = ((n[1]-1)*d[1], (n[2]-1)*d[2])

# Velocity [km/s]
v =  1f0./sqrt.(m);
idx_wb = minimum(find_water_bottom(v.-v[1,1]))
function gaussian_background(m::Matrix{Float32}, width::Union{Number, Tuple{Number, Number}}; idx_wb=idx_wb)
    v =  1f0./sqrt.(m);
    v0 = deepcopy(v);
    v0[:,idx_wb+1:end] = imfilter(v[:,idx_wb+1:end], Kernel.gaussian(width))
    m0 = (1f0 ./ v0).^2
    return m0
end

# background model
init_width = 10;
m0 = gaussian_background(m, init_width);

# Setup info and model structure
nsrc = 16	# number of sources
model = Model(n, d, o, m; nb=40)
model0 = Model(n, d, o, m0; nb=40)

# Set up receiver geometry
nxrec = n[1]
xrec = range(0f0, stop=(n[1] -1)*d[1], length=nxrec)
yrec = 0f0
#zrec = range((idx_wb-1)*d[2], stop=(idx_wb-1)*d[2], length=nxrec)
zrec = range(10f0, stop=10f0, length=nxrec)

# receiver sampling and recording time
timeD = 1800f0   # receiver recording time [ms]
dtD = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)
# Set up source geometry (cell array with source locations for each shot)
xsrc = range(0f0, stop=(n[1] -1)*d[1], length=nsrc)
ysrc = range(0f0, stop=0f0, length=nsrc)
zsrc = range(10f0, stop=10f0, length=nsrc)

# Set up source structure
srcGeometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dtD, t=timeD)
# setup wavelet
f0 = 0.025f0     # kHz
wavelet = ricker_wavelet(timeD, dtD, f0)
q = judiVector(srcGeometry, wavelet)

###################################################################################################
opt = Options(isic=true)

# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F(model0), q)

# Nonlinear modeling
@time dobs = F * q
@time rtm = J'*dobs

# Preconditioners
Mr = judiTopmute(n, idx_wb, 1) * judiDepthScaling(model0) * judiDepthScaling(model0)

exp_config = @strdict nsrc nxrec n d init_width
fig = figure(figsize=(20,12))
plot_simage(reshape(Mr * rtm, n)', d; d_scale=0, new_fig=false, name="initial RTM")
tight_layout();
safesave(joinpath(plotsdir("compass-example"),savename(exp_config; digits=6)*"_init.png"), fig); 
close(fig);

for i = 1:10
    m0 = gen_m0_vary(m)

    fig = figure(figsize=(20,12))
    plot_velocity(m0', d; new_fig=false, name="background model", vmax=maximum(m))
    tight_layout();
    safesave(joinpath(plotsdir("compass-example"),savename(exp_config; digits=6)*"_background.png"), fig); 
    close(fig);
end

for i = 1:5

    exp_config = @strdict nsrc nxrec n d

    m0 = gen_m0_vary(m)

    J = judiJacobian(F(; m=m0), q)
    @time rtm = J'*dobs

    fig = figure(figsize=(20,12))
    plot_simage(reshape(Mr * rtm, n)', d; d_scale=0, new_fig=false, name="RTM")
    tight_layout();
    safesave(joinpath(plotsdir("compass-example"),savename(exp_config; digits=6)*"_continued.png"), fig); 
    close(fig);
    
end
