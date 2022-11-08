# author: Ziyi Yin, ziyi.yin@gatech.edu
# Nov 2022
# This script trains a Fourier Neural Operator that does velocity continuation

using DrWatson
@quickactivate "Amortized-FNO-VC"

using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using Statistics, LinearAlgebra
using CUDA
using ProgressMeter
using InvertibleNetworks:ActNorm
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

# Define raw data directory
continued_background_path = datadir("background-models", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2")
init_rtm_path = datadir("init", "nslice=$(nslice)_nsrc=$(nsrc).jld2")
continued_rtm_path = datadir("rtms", "lengthmax=$(lengthmax)_ncont=$(ncont)_nslice=$(nslice).jld2")

mkpath(datadir("background-models"))
mkpath(datadir("init"))
mkpath(datadir("rtms"))
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
function get_train_valid()
    continued_background_dict = JLD2.load(continued_background_path)["m0set"]
    init_m0 = JLD2.load(init_rtm_path)["m0_init_set"];
    init_rtm = JLD2.load(init_rtm_path)["rtm_init_set"];
    continued_rtm_set = JLD2.load(continued_rtm_path)["rtmset"];

    continued_m0_set = zeros(Float32, 650, 341, nsample);
    for i = 1:nsample
        continued_m0_set[:,:,i] = continued_background_dict[i]
    end
    init_m0_set = repeat(init_m0, inner= [1, 1, ncont]);
    init_rtm_set = repeat(init_rtm, inner= [1, 1, ncont]);

    ## X and Y
    # scale RTM by 2000
    X = cat(reshape(init_m0_set, 650, 341, 1, nsample), reshape(init_rtm_set/2f3, 650, 341, 1, nsample), reshape(continued_m0_set, 650, 341, 1, nsample), dims=3);
    # nx, ny, nc, nsample
    Y = continued_rtm_set/2f3;

    x_train  = cat(X[:,:,:,1:3200], reverse(X[:,:,:,1:3200], dims=1), dims=4);
    x_valid = X[:,:,:,3200+1:3200+600];

    y_train  = cat(Y[:,:,1:3200], reverse(Y[:,:,1:3200], dims=1), dims=3);
    y_valid = Y[:,:,3200+1:3200+600];

    return x_train, x_valid, y_train, y_valid
end

x_train, x_valid, y_train, y_valid = get_train_valid();

## n,d 
n = (325, 341)
d = 1f0 ./ n

## grid
grid = gen_grid(n, d);

ntrain = size(x_train)[end] * 4         #windowing
nvalid = size(x_valid)[end] * 4         #windowing

## network structure
batch_size = 20
learning_rate = 2f-3
epochs = 5000
modes = 96
width = 32

AN = ActNorm(ntrain)
AN.forward(x_train);

NN = Net2d(modes, width; in_channels=5, out_channels=1, mid_channels=128)
gpu_flag && (global NN = NN |> gpu)

Flux.trainmode!(NN, true)
w = Flux.params(NN)

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)
nbatches = Int(floor(ntrain/batch_size))
nbatches_valid = Int(floor(nvalid/batch_size))

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)
prog = Progress(ntrain * epochs)

# plot figure
x_plot = x_valid[1:325, :, :, 1]
y_plot = y_valid[1:325, :, 1]

x_plot_train = x_train[1:325, :, :, 1]
y_plot_train = y_train[1:325, :, 1]

# Define result directory

sim_name = "2D-FNO-vc-compass"
exp_name = "velocity-continuation"

save_dict = @strdict exp_name
plot_path = plotsdir(sim_name, savename(save_dict; digits=6))

## training
println("start training")
window_ = true
for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain)[1:batch_size*nbatches], batch_size, nbatches)
    idx_v = reshape(randperm(nvalid)[1:batch_size*nbatches_valid], batch_size, nbatches_valid)

    Flux.trainmode!(NN, true)
    for b = 1:nbatches
        img_idx = (idx_e[:,b] .- 1) .÷ 4
        window_idx = (idx_e[:,b] .- 1) .% 4 .+ 1
        x_window = cat([window(x_train[:, :, :, img_idx[i]:img_idx[i]], window_idx[i]) for i = 1:length(img_idx)]..., dims=4)
        x = tensorize(x_window, grid, AN)
        y = cat([window(y_train[:, :, img_idx[i]:img_idx[i]], window_idx[i]) for i = 1:length(img_idx)]..., dims=3)
        if gpu_flag
            x = x |> gpu
            y = y |> gpu
        end
        grads = gradient(w) do
            global loss = norm(NN(x)-y)/norm(y)      # NN matches the residual
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
        ProgressMeter.next!(prog; showvalues = [(:loss, loss), (:epoch, ep), (:batch, b)])
    end

    Flux.testmode!(NN, true)
    NN_save = NN |> cpu
    @time y_predict = NN_save(tensorize(reshape(x_plot,n[1],n[2],3,1), grid, AN))
    @time y_predict_train = NN_save(tensorize(reshape(x_plot_train,n[1],n[2],3,1), grid, AN))

    ### plot training
    fig = figure(figsize=(16, 20))

    subplot(5,2,1)
    plot_velocity(x_plot_train[:,:,1]', (6f0, 6f0); new_fig=false, vmax=0.25, name="initial background model", cmap="GnBu"); colorbar();
    
    subplot(5,2,2)
    plot_simage(x_plot_train[:,:,2]', (6f0, 6f0); new_fig=false, cmap="Greys", name="initial RTM", vmax=0.3); colorbar();
    
    subplot(5,2,3)
    plot_velocity(x_plot_train[:,:,3]', (6f0, 6f0); new_fig=false, vmax=0.25, name="new background model", cmap="GnBu"); colorbar();
    
    subplot(5,2,4)
    plot_simage(y_predict_train[:,:,1]', (6f0, 6f0); new_fig=false, cmap="Greys", name="predicted continued RTM", vmax=0.3); colorbar();
    
    subplot(5,2,5)
    plot_simage(y_plot_train', (6f0, 6f0); new_fig=false, cmap="Greys", name="true continued RTM", vmax=0.3); colorbar();
    
    subplot(5,2,6)
    plot_simage(y_predict_train[:,:,1]'-y_plot_train', (6f0, 6f0); new_fig=false, cmap="Greys", vmax=0.3, name="diff"); colorbar();
    
    subplot(5,2,7)
    plot(x_plot_train[:,:,2][167,:]);
    plot(y_plot_train[167,:]);
    legend(["initial", "true"])
    title("vertical profile at 1km")

    subplot(5,2,8)
    plot_simage(x_plot_train[:,:,2]'-y_plot_train', (6f0, 6f0); new_fig=false, cmap="Greys", vmax=0.3, name="diff between initial RTM and continued RTM"); colorbar();
    
    subplot(5,2,9)
    plot(y_predict_train[167,:,1]);
    plot(y_plot_train[167,:]);
    legend(["predict","true"])
    title("vertical profile at 1km")
    
    subplot(5,2,10)
    plot(y_predict_train[333,:,1]);
    plot(y_plot_train[333,:]);
    legend(["predict","true"])
    title("vertical profile at 2km")

    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_train.png"), fig);
    close(fig)

    ### plot validation
    fig = figure(figsize=(16, 20))

    subplot(5,2,1)
    plot_velocity(x_plot[:,:,1]', (6f0, 6f0); new_fig=false, vmax=0.25, name="initial background model", cmap="GnBu"); colorbar();
    
    subplot(5,2,2)
    plot_simage(x_plot[:,:,2]', (6f0, 6f0); new_fig=false, cmap="Greys", name="initial RTM", vmax=0.3); colorbar();
    
    subplot(5,2,3)
    plot_velocity(x_plot[:,:,3]', (6f0, 6f0); new_fig=false, vmax=0.25, name="new background model", cmap="GnBu"); colorbar();
    
    subplot(5,2,4)
    plot_simage(y_predict[:,:,1]', (6f0, 6f0); new_fig=false, cmap="Greys", name="predicted continued RTM", vmax=0.3); colorbar();
    
    subplot(5,2,5)
    plot_simage(y_plot', (6f0, 6f0); new_fig=false, cmap="Greys", name="true continued RTM", vmax=0.3); colorbar();
    
    subplot(5,2,6)
    plot_simage(y_predict[:,:,1]'-y_plot', (6f0, 6f0); new_fig=false, cmap="Greys", vmax=0.3, name="diff"); colorbar();
    
    subplot(5,2,7)
    plot(x_plot[:,:,2][167,:]);
    plot(y_plot[167,:]);
    legend(["initial", "true"])
    title("vertical profile at 1km")

    subplot(5,2,8)
    plot_simage(x_plot[:,:,2]'-y_plot', (6f0, 6f0); new_fig=false, cmap="Greys", vmax=0.3, name="diff between initial RTM and continued RTM"); colorbar();
    
    subplot(5,2,9)
    plot(y_predict[167,:,1]);
    plot(y_plot[167,:]);
    legend(["predict","true"])
    title("vertical profile at 1km")
    
    subplot(5,2,10)
    plot(y_predict[333,:,1]);
    plot(y_plot[333,:]);
    legend(["predict","true"])
    title("vertical profile at 2km")

    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_2Dfno_vc_valid.png"), fig);
    close(fig)

    img_idx = (idx_v[:,1] .- 1) .÷ 4
    window_idx = (idx_v[:,1] .- 1) .% 4 .+ 1
    x_window = cat([window(x_valid[:, :, :, img_idx[i]:img_idx[i]], window_idx[i]) for i = 1:length(img_idx)]..., dims=4)
    x_valid_e = tensorize(x_window, grid, AN)
    y_valid_e = cat([window(y_valid[:, :, img_idx[i]:img_idx[i]], window_idx[i]) for i = 1:length(img_idx)]..., dims=3)

    if gpu_flag
        x_valid_e = x_valid_e |> gpu
        y_valid_e = y_valid_e |> gpu
    end
    Loss_valid[ep] = norm(NN(x_valid_e) - y_valid_e)/norm(y_valid_e)

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

    w_save = Flux.params(NN_save)    

    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid loss_train loss_valid
    @tagsave(
        datadir(sim_name, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
    
end

NN_save = NN |> cpu
w_save = params(NN_save)

final_dict = @strdict Loss Loss_valid epochs NN_save w_save batch_size Loss modes width learning_rate epochs n d AN ntrain nvalid

@tagsave(
    datadir(sim_name, savename(final_dict, "jld2"; digits=6)),
    final_dict;
    safe=true
)
