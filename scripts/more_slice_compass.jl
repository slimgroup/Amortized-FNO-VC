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
using SegyIO
using ArgParse
Random.seed!(2022)

include(srcdir("utils.jl"))
parsed_args = parse_commandline()
block = segy_read(datadir("slim.gatech.edu/data/synth/Compass/final_velocity_model_ieee_6m.sgy"))

# original compass model is in 25m*25m*6m
n1 = (1911,2730,341)
d = (25f0,25f0,6f0)

n = (637,910,341)

sx = get_header(block, "SourceX")
sy = get_header(block, "SourceY")

v_nogas = zeros(Float32,n)
v_gas = zeros(Float32,n)

for i = 1:n[1]
    x = d[1].*(i-1)
    inds = findall(sx.==x)
    slice = block.data[:,inds[sortperm(sy[inds])]]

    v_nogas[i,:,:] = transpose(slice[:,1:Int(end/3)])
end

vtotal = Float32.(imresize(v_nogas, n1))./1f3;

xshift = Int.(round.(range(700, stop=size(vtotal,1)-651, length=12)))
yshift = Int.(round.(range(1, stop=size(vtotal,1)-651, length=12)))
yshift = 1:200:size(vtotal,2)

xshift = 700:size(vtotal,1)-651
yshift = 300:size(vtotal,1)-651
vset = zeros(Float32, 650, 341, 20)
for i = 1:20
    xrand = rand(xshift)
    yrand = rand(yshift)
    vset[:,:,i] = vtotal[xrand:649+xrand,yrand,:];
    figure();imshow(vset[:,:,i]'); title("xrand=$xrand,yrand=$yrand")
end
JLD2.@save datadir("velocity_set.jld2") vset