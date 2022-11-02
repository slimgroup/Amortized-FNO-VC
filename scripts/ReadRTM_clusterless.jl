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

nwk = parse(Int, AzureClusterlessHPC.__params__["_NODE_COUNT_PER_POOL"])

# Setup AzStorage
session = AzSession(;protocal=AzClientCredentials, resource=registry_info["_RESOURCE"])
container = AzContainer("amortized-fno-vc"; storageaccount=registry_info["_STORAGEACCOUNT"], session=session)
mkpath(container)

function read_i(i, container)
    idx, m0, rtm = deserialize(container,  "rtm_sample$(i)")
    return idx, m0, rtm
end

nsample = nslice * ncont

rtmset = zeros(Float32, n[1], n[2], nsample);

v = vset[:,:,1]
m = 1f0./v.^2f0
model = Model(n, d, o, m; nb=80)
idx_wb = 35
Mr = judiTopmute(n, idx_wb, 1) * judiDepthScaling(model) * judiDepthScaling(model)

for i = 1:nsample
    idx, m0, rtm = deserialize(container,  "rtm_sample$(i)")
    rtmset[:,:,i] = reshape(Mr * vec(rtm.data), n)
end

rtm_dict = @strdict lengthmax rtmset ncont nslice
@tagsave(
    datadir("rtms", savename(rtm_dict, "jld2"; digits=6)),
    rtm_dict;
    safe=true
)
