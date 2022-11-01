# Author: Ziyi Yin
# Date: October 2022
#

using DrWatson
@quickactivate "Amortized-FNO-VC"

# Load package
using JUDI, HDF5, PyPlot, Images, PyCall, Random, JOLI, Statistics, JSON
using AzStorage, AzSessions, LinearAlgebra, Serialization
using ArgParse
using JLD2
using Distributions
Random.seed!(2022)

include(srcdir("utils.jl"));
JLD2.@load datadir("velocity_set.jld2") vset;
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
if ~isfile(datadir("background-models", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2"))
    m0set = vcat([[gen_m0_vary(mset[:,:,j]; lengthmax=lengthmax) for i = 1:ncont] for j = 1:nslice]...)

    m0_dict = @strdict lengthmax m0set ncont nslice
    @tagsave(
        datadir("background-models", savename(m0_dict, "jld2"; digits=6)),
        m0_dict;
        safe=true
    )
else
    JLD2.@load datadir("background-models", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2") m0set
end

## load set of data
JLD2.@load datadir("seismic-data", "nslice=$(nslice)_nsrc=$nsrc.jld2") dobs_set
d_obs_set = repeat(dobs_set, inner=ncont)

# Set paths to credentials + parameters
# Use Standard_F4 or something that has enough memeory but cheap

ENV["CREDENTIALS"] = joinpath(pwd(), "../credentials.json")
ENV["PARAMETERS"] = joinpath(pwd(), "../params.json")
ENV["REGISTRYINFO"] = joinpath(pwd(), "../registryinfo.json")

registry_info = JSON.parsefile(ENV["REGISTRYINFO"])

using AzureClusterlessHPC
batch_clear();

# Azure continer
batch = pyimport("azure.batch")
container_registry = batch.models.ContainerRegistry(registry_server=registry_info["_REGISTRY_SERVER"],
                                                     user_name=registry_info["_USER_NAME"],
                                                     password=registry_info["_PASSWORD"])

# Autoscale formula
auto_scale_formula = """
    samples = \$PendingTasks.GetSamplePercent(TimeInterval_Minute * 15);
    tasks = samples < 10 ? max(0,\$PendingTasks.GetSample(1)) : max(\$PendingTasks.GetSample(1), avg(\$PendingTasks.GetSample(TimeInterval_Minute * 15)));
    targetVMs = tasks > 0? tasks:max(0, \$TargetDedicatedNodes/2);
    \$TargetDedicatedNodes=max(0, min(targetVMs, 4));
    \$NodeDeallocationOption = taskcompletion;"""

    nwk = parse(Int, AzureClusterlessHPC.__params__["_NODE_COUNT_PER_POOL"])
create_pool(container_registry=container_registry; enable_auto_scale=true, auto_scale_formula=auto_scale_formula)

# Setup AzStorage
session = AzSession(;protocal=AzClientCredentials, resource=registry_info["_RESOURCE"])
container = AzContainer("amortized-fno-vc"; storageaccount=registry_info["_STORAGEACCOUNT"], session=session)
mkpath(container)

@batchdef using JUDI, AzStorage, Distributed, AzureClusterlessHPC, Serialization, LinearAlgebra, Statistics, Images, Distributions
    
@batchdef function mute_turning(d_obs::judiVector{Float32, Matrix{Float32}}, q::judiVector{Float32, Matrix{Float32}}; t0::Number=1f-1, v_water::Number=1480f0)
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

# ocean bottom
@batchdef idx_wb = 35

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
opt = JUDI.Options(isic=true)

@batchdef begin
    d = (6f0, 6f0);
    o = (0f0, 0f0);
    n = (650, 341);
end
@batchdef function rtm(i, q, m0, dobs, container)

    d = (6f0, 6f0);
    o = (0f0, 0f0);
    n = (650, 341);

    ## mute turning wave
    dobs = mute_turning(dobs, q)

    # Setup info and model structure
    model0 = Model(n, d, o, m0; nb=80)

    # Setup operators
    F = judiModeling(model0, q.geometry, dobs.geometry; options=JUDI.Options(isic=true))
    J = judiJacobian(F, q)
    @time rtm = J'*dobs

    # container
    serialize(container, "rtm_sample$(i)", (i=i, m0=m0, rtm=rtm))
    return nothing
end

println("RTM")
Base.flush(stdout)

nsample = nslice * ncont
nsample = 50

@batchdef nslice = 20
@batchdef ncont = 200

futures = @batchexec pmap(i -> rtm(i, q, m0set[i], d_obs_set[i], container), 1:nsample)
fetch(futures; num_restart=0, timeout=24000, task_timeout=1000)

delete_all_jobs()
delete_pool()
delete_container()
