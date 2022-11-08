# Author: Ziyi Yin
# Date: October 2022
#

using DrWatson
@quickactivate "Amortized-FNO-VC"
using JUDI, LinearAlgebra, Images, PyPlot
using JLD2
using SlimPlotting
using Random
using Distributions
using ArgParse
using InvertibleNetworks:ActNorm
Random.seed!(2022)

include(srcdir("utils.jl"));
JLD2.@load datadir("velocity_set_2.jld2") vset;
d = (6f0, 6f0);
o = (0f0, 0f0);
n = (650, 341);
extent = ((n[1]-1)*d[1], (n[2]-1)*d[2])

## args input
parsed_args = parse_commandline()
nsrc = parsed_args["nsrc"]
lengthmax = parsed_args["length"]
ncont = parsed_args["ncont"]

# ocean bottom
idx_wb = 35

# Set up receiver geometry
nxrec = n[1]
xrec = range(0f0, stop=(n[1] -1)*d[1], length=nxrec)
yrec = 0f0
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

## number of slices
nslice = size(vset)[end]

dobs_set = Vector{judiVector{Float32, Matrix{Float32}}}(undef, nslice)
for i = 1:nslice
    v = vset[:,:,i]
    m = 1f0./v.^2f0
    m0 = gaussian_background(m, 20);

    # Setup info and model structure
    model = Model(n, d, o, m; nb=80)
    model0 = Model(n, d, o, m0; nb=80)

    # Setup operators
    F = judiModeling(model, srcGeometry, recGeometry; options=opt)
    J = judiJacobian(F(model0), q)

    # Nonlinear modeling
    @time dobs_set[i] = F * q
end

save_dict = @strdict nslice nsrc dobs_set
@tagsave(
    joinpath(datadir("seismic-data-2"), savename(save_dict, "jld2"; digits=6)),
    save_dict;
    safe=true
)

