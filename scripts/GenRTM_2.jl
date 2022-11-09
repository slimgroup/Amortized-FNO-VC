# Author: Ziyi Yin
# Date: October 2022
#

using DrWatson
@quickactivate "Amortized-FNO-VC"

# Load package
using JUDI, HDF5, PyPlot, Images, PyCall, Random, JOLI, Statistics, JSON
using LinearAlgebra
using ArgParse
using JLD2
using Distributions
using InvertibleNetworks:ActNorm
Random.seed!(2022)

include(srcdir("utils.jl"));
JLD2.@load datadir("velocity_set_2.jld2") vset;
mset = 1f0./vset.^2f0;
d = (6f0, 6f0);
o = (0f0, 0f0);
n = (650, 341);
extent = ((n[1]-1)*d[1], (n[2]-1)*d[2])

## args input
parsed_args = parse_commandline()
nsrc = parsed_args["nsrc"]
lengthmax = parsed_args["length"]
ncont = parsed_args["ncont"]
nslice = size(vset)[end]

## generate a set of m0
idx_wb = 35
if ~isfile(datadir("background-models-2", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2"))
    m0set = vcat([[gen_m0_vary(mset[:,:,j]; lengthmax=lengthmax) for i = 1:ncont] for j = 1:nslice]...)

    m0_dict = @strdict lengthmax m0set ncont nslice
    @tagsave(
        datadir("background-models-2", savename(m0_dict, "jld2"; digits=6)),
        m0_dict;
        safe=true
    )
else
    JLD2.@load datadir("background-models-2", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2") m0set
end

## load set of data
JLD2.@load datadir("seismic-data-2", "nslice=$(nslice)_nsrc=$nsrc.jld2") dobs_set
d_obs_set = repeat(dobs_set, inner=ncont)

# ocean bottom
idx_wb = 35

# Set up receiver geometry
nxrec = n[1]
xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
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

function rtm(i, q, m0, dobs)

    ## mute turning wave
    dobs = mute_turning(dobs, q)

    # Setup info and model structure
    model0 = Model(n, d, o, m0; nb=80)

    # Setup operators
    F = judiModeling(model0, q.geometry, dobs.geometry; options=JUDI.Options(isic=true))
    J = judiJacobian(F, q)
    @time rtm_i = J'*dobs
    Mr = judiTopmute(n, idx_wb, 1) * judiDepthScaling(model0) * judiDepthScaling(model0)

    # container
    return reshape(Mr * rtm_i, n)
end

println("RTM")
Base.flush(stdout)

nsample = nslice * ncont
counter = 0
for j = 6:nslice
    global rtmset = zeros(Float32, n[1], n[2], ncont)
    for i = 1:ncont
        global counter = counter + 1
        Base.flush(stdout)
        println("sample $counter")
        global rtmset[:,:,i] = rtm(i, q, m0set[counter], d_obs_set[counter])
    end
    rtm_dict = @strdict lengthmax rtmset ncont nslice
    @tagsave(
        datadir("rtms-2-slice-$j", savename(rtm_dict, "jld2"; digits=6)),
        rtm_dict;
        safe=true
    )
end

