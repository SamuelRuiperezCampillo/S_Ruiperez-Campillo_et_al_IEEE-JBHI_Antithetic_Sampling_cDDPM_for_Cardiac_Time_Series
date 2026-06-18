"""Training/sweep driver for the convolutional and fully-connected beta-VAE.

This module is the original VAE training entry point: it sweeps a list of KL
weights ``beta``, trains the selected ``AutoEncoder`` (CNN or FC) on the
ventricular MAP dataset with the ELBO objective (MSE reconstruction + beta * KL),
logs to Weights & Biases, plots reconstructions, and dumps a per-run summary to
JSON. This is a faithful migration of the original ``vae.py`` into the package
layout; the only changes are import rewrites to the package and routing of
hard-coded local filesystem paths through ``cardiac_map_diffusion.paths``. The
training loop, loss definition, hyperparameters, RNG behaviour, and forward-pass
math are unchanged.
"""

import torch
import matplotlib.pyplot as plt
from cardiac_map_diffusion.data.retrieve_dataset import NumpyDataSet
from cardiac_map_diffusion.data.data_baselines import get_train_test
import os
import numpy as np
import matplotlib as mpl
import wandb
import json
from cardiac_map_diffusion import paths
# % Import ad hoc modules
# TODO(paths): source-code location for ad hoc modules; no paths accessor fits a
# source tree, and this value is unused below. Original literal preserved.
path_modules = os.path.join(r'C:/Users/sruip/Desktop/Universities/ETH',
                            r'Research_Projects/Master_Thesis/MAP_autoencoder')



main_path_VAE = str(paths.experiments_root())
learning_rate = 0.0005
num_epochs = 15
batch_size = 16
architecture = 6
architectures = [0, 1, 2, 3, 4, 5, 6]
architectures = [6]
model = 'CNN'
if model == 'CNN':
    from cardiac_map_diffusion.models.autoencoder_conv import AutoEncoder
elif model == 'FC':
    from cardiac_map_diffusion.models.autoencoder import AutoEncoder
#betas = [0, 1e-5, 1e-6, 5e-7, 2e-7, 1.5e-7, 1e-7, 9.5e-8, 9e-8, 8e-8, 5e-8, 1e-8, 1e-9]
#betas = [1e-7]
betas = [5, 2, 1, 0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 0]
beta = 0
all_summary = dict()
count = 0
for beta in betas:
    count = count+1
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="VAE_pytorch",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "VAE_conv",
            "dataset": "ventricular_MAP",
            "epochs": num_epochs,
            "beta": beta,
            "architecture": architecture
        },
        mode="online"
    )


    os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
    X_std_train, X_std_test = get_train_test()

    train = NumpyDataSet(X_std_train)
    test = NumpyDataSet(X_std_test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)

    input_dim = X_std_train.shape[-1]
    latent_size = 32
    vae = AutoEncoder(input_dim, latent_size, architecture=architecture).float()

    # %%
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
    #mse_loss = torch.nn.MSELoss(reduction="mean") # None (or sum/batch_size)
    mse_loss = torch.nn.MSELoss(reduction="sum")  # None (or sum/batch_size)
    # mse

    def loss_fn(x, recon_x, mean, log_var, beta=1):
        # Generalised version
        # prior_distribution = torch.distributions.normal.Normal(prior_mu, prior_log_sigma.exp()+1e-6)
        # posterior_distribution = torch.distributions.normal.Normal(mu, log_sigma.exp()+1e-6)
        # z_kl_div = torch.distributions.kl.kl_divergence(posterior_distribution, prior_distribution).sum()
        batch_size = len(x)
        # Closed form
        kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).sum() # scalar
        #recon_loss = bce_loss(recon_x, x)
        recon_loss = mse_loss(recon_x.float(), x.float())
        #weight_recons = 1
        #elbo = weight_kl*kld_loss + recon_loss
        #elbo = torch.mean(recon_loss + (beta * kld_loss))
        elbo = (recon_loss + (beta * kld_loss))/batch_size
        return elbo, kld_loss, recon_loss


    elbo_w_neg = []
    kl_div = []
    kl_div_w = []
    loss_train = []
    loss_test = []

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_kld = 0.0
        running_bce = 0.0
        running_bce_test = 0.0
        num_batches = 0

        for i, x in enumerate(train_loader):
            optimizer.zero_grad()

            # forward + backward + optimize
            if model == 'CNN':
                recon_x, mean, log_var = vae(x.float(), architecture=architecture)
            elif model == 'FC':
                recon_x, mean, log_var = vae(x.float())
            loss, kld_loss, recon_loss = loss_fn(x, recon_x, mean, log_var, beta)
            loss.backward()
            optimizer.step()

            # Test tracking
            for i_test, x_test in enumerate(test_loader):
                with torch.no_grad():
                    if model == 'CNN':
                        recon_x_test, mean_test, log_var_test = vae(x_test.float(), architecture=architecture)
                    elif model == 'FC':
                        recon_x_test, mean_test, log_var_test = vae(x_test.float())
                    _, _, recon_loss_test = loss_fn(x_test, recon_x_test,
                                                    mean_test, log_var_test, beta)
                    running_bce_test += recon_loss_test.item()

            # print statistics
            running_loss += loss.item()
            running_kld += kld_loss.item()
            running_bce += recon_loss.item()
            num_batches += 1

        elbo_w_neg.append(-running_loss / num_batches)
        kl_div.append(running_kld / num_batches)
        kl_div_w.append(beta * running_kld / num_batches)
        loss_train.append(running_bce / num_batches)
        loss_test.append(recon_loss_test.item() / num_batches)

        wandb.log({"-ELBO_w": -running_loss / num_batches,
                   "KL-D": running_kld / num_batches,
                   #"BCE": running_bce / num_batches,
                   #"BCE_test": recon_loss_test / num_batches,
                   #"BCE_weighted": 0.01 * running_bce / num_batches,
                   "MSE": running_bce / num_batches,
                   "MSE_test": recon_loss_test / num_batches,
                   "KL-D weighted": beta * running_kld / num_batches})
        print(f"Epoch: {epoch}, -ELBO: {-running_loss / num_batches}, KL-D: {running_kld / num_batches}, BCE: {running_bce / num_batches}")

    print('Finished Training')

    summary_dict = dict()
    summary_dict = {
        'architecture': 'FC_NN',
        'architecture_type': architecture,
        'loss_type': 'mse',
        'n_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'latent_size': latent_size,
        'beta': beta,
        'elbo_neg': elbo_w_neg,
        'kl_div': kl_div,
        'kl_div_w': kl_div_w,
        'loss_train': loss_train,
        'loss_test': loss_test,
    }
    all_summary[f"model_{count}"] = summary_dict



    # plt.imshow(X_recon, cmap='Greys')
    def plot_MAP_vs_reconstructed(i, MAP_original, recons, dpi=600):
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
        })
        recons = np.array(recons)
        recons_av = np.mean(recons, axis=0)
        recons_std = np.std(recons, axis=0)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        plt.plot(MAP_original, color='k', linewidth=1.7)
        plt.plot(recons_av, color='b', linewidth=1.7)
        plt.ylabel('Voltage Amplitude [mV] (normalised)', size=15)
        plt.xlabel('Time [msec]', size=15)
        plt.grid(visible=bool, which='major', axis='both', color='gray', linestyle='--',
                 linewidth=0.6)
        plt.minorticks_on()
        ax.legend(['Original', 'Reconstructed'])
        plt.grid(visible=bool, which='minor', axis='both', color='gray', linestyle='--',
                 linewidth=0.4)
        plt.fill_between(range(len(recons_av)), recons_av + recons_std, recons_av - recons_std,
                         alpha=0.2, edgecolor='b', facecolor='b')
        plt.title(f"MAP_VAE_reconstruction lr={learning_rate}, ne={num_epochs}, beta={beta}")
        # plt.grid(b=bool, which='minor', axis='both', color='gray', linestyle='--',
        #         linewidth=0.4)
        fig.savefig(os.path.join(main_path_VAE, f"out_{i}_arch_{str(architecture)}_beta_{str(beta)}.png"))


    te = [0, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]

    for i in te:
        recons = []
        for k in range(50):
            with torch.no_grad():
                for i_test, x_test in enumerate(test_loader):
                    recon_x, _, _ = vae(x_test[i].unsqueeze(0).float(), architecture=architecture)
                #recon_x = torch.sigmoid(recon_x)
                recons.append(recon_x.detach().numpy().squeeze())

        plot_MAP_vs_reconstructed(i, X_std_test[i], recons)

    wandb.finish()


with open(os.path.join(main_path_VAE, "summary_models_dict.json"), "w") as fp:
    json.dump(all_summary, fp, indent=4)  # encode dict into JSON
print("Done writing dict into .json file")