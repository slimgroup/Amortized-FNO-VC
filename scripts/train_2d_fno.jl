# author: Ziyi Yin, ziyi.yin@gatech.edu
# Nov 2022
# This script trains a Fourier Neural Operator that does velocity continuation

using DrWatson
@quickactivate "Amortized-FNO-VC"

using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using MAT, Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
using HDF5
using SlimPlotting
using ArgParse
using JUDI
matplotlib.use("Agg")

Random.seed!(1234)
include(srcdir("utils.jl"))

## args input
parsed_args = parse_commandline()
nsrc = parsed_args["nsrc"]
lengthmax = parsed_args["length"]
ncont = parsed_args["ncont"]
nslice = parsed_args["nslice"]
nsample = nslice * ncont

## n,d 
n = (650, 341)
d = 1f0 ./ n
ntrain = 1600
nvalid = 300

# Define raw data directory
continued_background_path = datadir("background-models", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2")
init_rtm_path = datadir("init", "nslice=$(nslice)_nsrc=$(nsrc).jld2")
continued_rtm_path = datadir("rtms", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2")

if ~isfile(continued_background_path)
    run(`wget https://www.dropbox.com/s/8e3q0j414peeh83/'
        'lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2 -q -O $continued_background_path`)
end
if ~isfile(init_rtm_path)
    run(`wget https://www.dropbox.com/s/7fjzpqawtlylqh5/'
    'nslice=$(nslice)_nsrc=$(nsrc).jld2 -q -O $init_rtm_path`)
end
if ~isfile(continued_rtm_path)
    run(`wget https://www.dropbox.com/s/ivjj7juuuxn6lz3/'
    'lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2 -q -O $continued_rtm_path`)
end

#load
continued_background_dict = JLD2.load(continued_background_path)
init_rtm_dict = JLD2.load(init_rtm_path)
continued_rtm_dict = JLD2.load(continued_rtm_path)

continued_m0_set = zeros(Float32, n[1], n[2], nsample);
for i = 1:nsample
    continued_m0_set[:,:,i] = continued_background_dict["m0set"][i]
end
init_m0 = init_rtm_dict["m0_init_set"];
init_m0_set = repeat(init_m0, inner= [1, 1, ncont]);
init_rtm = init_rtm_dict["rtm_init_set"];
init_rtm_set = repeat(init_rtm, inner= [1, 1, ncont]);
continued_rtm_set = continued_rtm_dict["rtmset"];

## grid
grid = gen_grid(n, d);

## X and Y
# scale RTM by 2000
X = cat(init_m0_set, init_rtm_set/2f3, continued_m0_set, dims=4);
X = permutedims(X, [1,2,4,3]);  # nx, ny, nc, nsample
Y = continued_rtm_set/2f3;

x_train = X[:,:,:,1:ntrain];
x_valid = X[:,:,:,ntrain+1:ntrain+nvalid];

y_train = Y[:,:,1:ntrain];
y_valid = Y[:,:,ntrain+1:ntrain+nvalid];

## network structure
batch_size = 10
learning_rate = 2f-3
epochs = 5000
modes = 24
width = 32

AN = ActNorm(ntrain)
AN.forward(x_train);

function tensorize(x::AbstractArray{Float32, 3},grid::Array{Float32,3},AN::ActNorm)
    # input nx*ny, output nx*ny*4*1
    nx, ny, _ = size(grid)
    return cat(AN(reshape(x, nx, ny, 3, 1))[:,:,:,1], reshape(grid, nx, ny, 2), dims=3)
end

tensorize(x::AbstractArray{Float32,4},grid::Array{Float32,3},AN::ActNorm) = cat([tensorize(x[:,:,:,i],grid,AN) for i = 1:size(x,4)]..., dims=4)

NN = Net2d(modes, width; in_channels=5, out_channels=1, mid_channels=128)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[:, :, :, 1]
y_plot = y_valid[:, :, 1]

# Define result directory

sim_name = "2D_FNO_vc"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain)[1:batch_size*nbatches], batch_size, nbatches)

    Flux.trainmode!(NN, true)
    for b = 1:nbatches
        x = tensorize(x_train[:, :, :, idx_e[:,b]], grid, AN)
        y = y_train[:, :, idx_e[:,b]]
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        grads = gradient(w) do
            global loss = norm(NN(x)-y)^2f0
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    Flux.testmode!(NN, true)
    y_predict = NN(tensorize(x_plot, grid, AN) |> gpu)   |> cpu

    fig = figure(figsize=(16, 12))

    subplot(4,2,1)
    plot_velocity(x_plot[:,:,1]', (6f0, 6f0); new_fig=false, vmax=0.25, name="initial background model", cmap="GnBu"); colorbar();
    
    subplot(4,2,2)
    plot_simage(x_plot[:,:,2]', (6f0, 6f0); new_fig=false, cmap="seismic", name="initial RTM", vmax=0.3); colorbar();
    
    subplot(4,2,3)
    plot_velocity(x_plot[:,:,3]', (6f0, 6f0); new_fig=false, vmax=0.25, name="new background model", cmap="GnBu"); colorbar();
    
    subplot(4,2,4)
    plot_simage(y_predict[:,:,1]', (6f0, 6f0); new_fig=false, cmap="seismic", name="predicted continued RTM", vmax=0.3); colorbar();
    
    subplot(4,2,5)
    plot_simage(y_plot', (6f0, 6f0); new_fig=false, cmap="seismic", name="true continued RTM", vmax=0.3); colorbar();
    
    subplot(4,2,6)
    plot_simage(y_predict[:,:,1]'-y_plot', (6f0, 6f0); new_fig=false, cmap="RdGy", vmax=0.06, name="diff"); colorbar();
    
    subplot(4,2,7)
    plot(y_predict[500,:,1]);
    plot(y_plot[500,:]);
    legend(["predict","true"])
    title("vertical profile at 3km")
    
    subplot(4,2,8)
    plot(y_predict[333,:,1]);
    plot(y_plot[333,:]);
    legend(["predict","true"])
    title("vertical profile at 2km")

    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc.png"), fig);
    close(fig)

    Loss_valid[ep] = norm((NN(tensorize(x_valid, grid, AN) |> gpu)) - (y_valid |> gpu))^2f0 * batch_size/nvalid

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid); 
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("iterations")
    ylabel("value")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig);

    NN_save = NN |> cpu
    w_save = Flux.params(NN_save)    

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid loss_train loss_valid nsamples
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
    
end

NN_save = NN |> cpu
w_save = params(NN_save)

final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid nsamples

@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)