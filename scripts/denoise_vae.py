"""Train the VAE denoising model across all k-folds (entry script).

Faithful migration of ``MAP_VAE/vae_denoising_allfolds.py`` (the root copy that
the ``submit_jobs.sh`` launcher invokes). Only mechanical edits were applied:
imports were rewritten to the ``cardiac_map_diffusion`` package layout and this
module docstring was added. All training/evaluation logic is byte-for-byte
unchanged. Run via the absl config-file mechanism, e.g.
``python scripts/denoise_vae.py --config=configs/config.py``.
"""
import torch
from cardiac_map_diffusion.data.retrieve_dataset import retrieveDataSet
from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data, get_train_test_kfolds
import os
import logging
import json
import numpy as np
import cardiac_map_diffusion.metrics.map_functions_baselines as mapf
#from MAP_functions import loss_fn
import wandb
from absl import app
from ml_collections import config_flags
from cardiac_map_diffusion.training.lr_scheduler import CosineAnnealingLRWarmup
from cardiac_map_diffusion.training.beta_scheduler import frange_cycle_linear, frange_cycle_cosine, frange_cycle_sigmoid
import time
import pandas as pd

def main(_):
    summary_dict_list = []
    summary_values = []

    start_time = time.time()
    config = _CONFIG.value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = config.seed
    torch.manual_seed(random_seed)
    random_seed_split = config.seed_split

    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    architecture = config.architecture
    noise_type = config.noise_type
    model = config.model
    beta = config.beta
    working_dir = config.working_dir
    cluster = config.cluster
    optimizer_type = config.optimizer
    optimizer_scheduler = config.optimizer_scheduler
    beta_schedule_mode = config.beta_schedule_mode
    beta_schedule_cycles = config.beta_schedule_cycles
    beta_schedule_ratio = config.beta_schedule_ratio
    w_constant = config.weight_constant
    n_folds = config.num_folds
    #s_number = config.split_number
    n_workers = config.num_workers
    pref_factor = config.prefetch_factor
    latent_size = config.latent_size
    #FIGURE_DICT = config.FIGURE_DICT

    for s_number in range(n_folds):
        # setup directories
        config.experiment_dir = os.path.join(working_dir, config.experiment_name)
        if not os.path.isdir(config.experiment_dir):
            os.mkdir(config.experiment_dir)

        # set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(config.experiment_dir, f"{config.experiment_name}.log"),
                         mode="a")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"Starting {config.experiment_name} on {device}")
        with open(os.path.join(config.experiment_dir, "config.json"), "w") as f:
            f.write(config.to_json(indent=4))

        MAP_vent_complete = get_MAP_vent_data(CLUSTER=cluster)
        
        # Hidden test set logic
        df_hidden = None
        if hasattr(config, 'exclude_patients_file') and config.exclude_patients_file:
            if os.path.exists(config.exclude_patients_file):
                logger.info(f"Loading excluded patients from {config.exclude_patients_file}...")
                try:
                    excluded_df = pd.read_csv(config.exclude_patients_file)
                    if 'pat_ID' in excluded_df.columns:
                        excluded_pats = excluded_df['pat_ID'].astype(str).unique()
                        MAP_vent_complete['pat_ID'] = MAP_vent_complete['pat_ID'].astype(str)
                        
                        df_hidden = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(excluded_pats)]
                        initial_len = len(MAP_vent_complete)
                        MAP_vent_complete = MAP_vent_complete[~MAP_vent_complete['pat_ID'].isin(excluded_pats)]
                        final_len = len(MAP_vent_complete)
                        
                        logger.info(f"Excluded {len(excluded_pats)} patients.")
                        logger.info(f"Dataframe reduced from {initial_len} to {final_len} samples.")
                        logger.info(f"Hidden test set has {len(df_hidden)} samples.")
                    else:
                        logger.warning(f"Warning: 'pat_ID' column not found in {config.exclude_patients_file}")
                except Exception as e:
                    logger.error(f"Error reading excluded patients file: {e}")
            else:
                logger.warning(f"Warning: Excluded patients file {config.exclude_patients_file} not found.")

        noise_params = mapf.find_noise_params(noise_type)
        if noise_type == 'ep' or noise_type == 'allmixed':
            arrays = mapf.get_np_noisearrays(MAP_vent_complete)
        else:
            arrays = []

        if model == 'CNN':
            from cardiac_map_diffusion.models.autoencoder_conv import AutoEncoder
        elif model == 'FC':
            from cardiac_map_diffusion.models.autoencoder import AutoEncoder
        elif model == 'CNNres':
            from cardiac_map_diffusion.models.autoencoder_convres import AutoEncoder

        if not cluster:
            wandb.init(
                # set the wandb project where this run will be logged
                project="VAE_pytorch",
                name=f"model_{model}_arch_{str(architecture)}_beta_{str(beta)}_ntype_{noise_type[:3]}",
                # track hyperparameters and run metadata
                config={
                    "learning_rate": learning_rate,
                    "architecture": "VAE_conv",
                    "dataset": "ventricular_MAP",
                    "epochs": num_epochs,
                    "beta": beta,
                    "architecture_model": architecture
                },
                mode="disabled"
            )

        os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

        # Acquire data
        X_train, X_test, y_train, y_test = get_train_test_kfolds(MAP_vent_complete,
                                                                 num_folds=n_folds,
                                                                 split_number=s_number,
                                                                 r_seed=random_seed_split,
                                                                 apd_label='APD30_gs')
        X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)
        train, test, X_std_test_noisy = retrieveDataSet(noise_type, noise_params, X_train, X_test,
                                                        X_std_train, X_std_test, arrays=arrays)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_workers, pin_memory=True, prefetch_factor=pref_factor)
        test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False,
                                                  num_workers=n_workers, pin_memory = True, prefetch_factor=pref_factor)
        
        # Prepare hidden test set
        hidden_loader = None
        if df_hidden is not None and not df_hidden.empty:
             X_hidden = np.array(df_hidden['MAP_segments'].tolist())
             X_std_hidden = mapf.normalize_EGM_array(X_hidden)
             # Reuse retrieveDataSet to get a noisy dataset for the hidden set
             # Pass X_hidden as both train and test args, since we only need the dataset object
             _, hidden_ds, _ = retrieveDataSet(noise_type, noise_params, X_hidden, X_hidden, 
                                             X_std_hidden, X_std_hidden, arrays=arrays)
             hidden_loader = torch.utils.data.DataLoader(hidden_ds, batch_size=len(hidden_ds), shuffle=False,
                                                       num_workers=n_workers, pin_memory=True, prefetch_factor=pref_factor)

        # Initialize beta schedule after train_loader is created (CRITICAL FIX)
        if beta_schedule_mode == 'linear':
            beta_schedule = frange_cycle_linear(num_epochs * len(train_loader), n_cycle=beta_schedule_cycles,
                                                ratio=beta_schedule_ratio)
        elif beta_schedule_mode == 'sigmoid':
            beta_schedule = frange_cycle_sigmoid(num_epochs * len(train_loader), n_cycle=beta_schedule_cycles,
                                                ratio=beta_schedule_ratio)
        elif beta_schedule_mode == 'cosine':
            beta_schedule = frange_cycle_cosine(num_epochs * len(train_loader), n_cycle=beta_schedule_cycles,
                                                ratio=beta_schedule_ratio)
        elif beta_schedule_mode == 'none':
            beta_schedule = np.ones(num_epochs * len(train_loader))

        input_dim = X_std_train.shape[-1]
        if model == 'CNN':
            vae = AutoEncoder(input_dim, latent_size, architecture=architecture, weight_constant=w_constant).float()
        elif model == 'FC':
            vae = AutoEncoder(input_dim, latent_size).float()
        elif model == 'CNNres':
            vae = AutoEncoder(input_dim, latent_size).float()
        vae = vae.to(device)

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(vae.parameters(), lr=learning_rate)

        if optimizer_scheduler:
            warmup_steps = int(0.1 * config.num_epochs) * (len(train_loader))
            total_steps = config.num_epochs * (len(train_loader))
            scheduler = CosineAnnealingLRWarmup(optimizer, T_max=total_steps, T_warmup=warmup_steps)

        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        #mse_loss = torch.nn.MSELoss(reduction="mean") # None (or sum/batch_size)
        mse_loss = torch.nn.MSELoss(reduction="sum")  # None (or sum/batch_size)

        def loss_fn(x, recon_x, mean, log_var, beta=1, latent_size=32):
            # Generalised version
            # prior_distribution = torch.distributions.normal.Normal(prior_mu, prior_log_sigma.exp()+1e-6)
            # posterior_distribution = torch.distributions.normal.Normal(mu, log_sigma.exp()+1e-6)
            # z_kl_div = torch.distributions.kl.kl_divergence(posterior_distribution, prior_distribution).sum()
            batch_size = len(x)
            # Closed form
            kld_loss = -0.5 * torch.sum(
                1 + log_var - mean.pow(2) - log_var.exp()).sum() / latent_size  # scalar (/latent size)
            # recon_loss = bce_loss(recon_x, x)
            recon_loss = mse_loss(recon_x.cpu().float(), x.cpu().float())  # Fixed: move x to CPU too
            # weight_recons = 1
            # elbo = weight_kl*kld_loss + recon_loss
            # elbo = torch.mean(recon_loss + (beta * kld_loss))
            elbo = (recon_loss + (beta * kld_loss)) / batch_size

            # To Do: Divide the Kl term by the number of latent dimensions
            # Fix the seeds to make the experiments reproducible.
            # In the beginning it may be easier to reduce the KL-div., so I can do beta-annealing (increase slowly)
            # so that I reduce the load of the KL-div. in the beginning, and regularise later.
            # Posterior collapse: the posterior q collapses to the prior.

            return elbo, kld_loss, recon_loss



        elbo_w_neg = []
        kl_div = []
        kl_div_w = []
        loss_train = []
        loss_test = []
        
        # Final metrics will be computed once after training (not per-epoch)
        pcorr_train = []
        pcorr_test = []
        rmse_train = []
        rmse_test = []
        psnr_train = []
        psnr_test = []
        mse_train = []
        mse_test = []
        spearman_train = []
        spearman_test = []
        snr_train = []
        snr_test = []
        dtw_train = []
        dtw_test = []
        lsd_train = []
        lsd_test = []
        nmae_range_train = []
        nmae_range_test = []
        nmae_l1_train = []
        nmae_l1_test = []
        nmae_mean_train = []
        nmae_mean_test = []

        beta_scheduled_track = []
        kl_beta_scheduled_track = []
        kld_scheduled_track = []
        average_gradient_norms_epoch = []

        iter_count = 0
        for epoch in range(num_epochs):

            running_loss = 0.0
            running_kld = 0.0
            running_kld_beta = 0.0
            running_bce = 0.0
            running_bce_test = 0.0
            num_batches_train = 0
            gradient_norms_iter = []

            for i, (x, x_noisy) in enumerate(train_loader):
                #print(i)
                optimizer.zero_grad()
                x = x.squeeze().to(device)  # Fixed: move x to device
                x_noisy = x_noisy.squeeze().to(device)

                # Fix shape for Batch Size 1 (Hidden Test Set)
                if x_noisy.ndim == 1:
                    x_noisy = x_noisy.unsqueeze(0) # Becomes (1, Length)
                
                # Check target shape matching
                if x.ndim == 1:
                     x = x.unsqueeze(0)

                # forward + backward + optimize
                if model == 'CNN':
                    recon_x, mean, log_var = vae(x_noisy.float(), architecture=architecture)
                elif model == 'FC':
                    recon_x, mean, log_var = vae(x_noisy.float())
                elif model == 'CNNres':
                    recon_x, mean, log_var = vae(x_noisy.float())
                multiplier = beta_schedule[iter_count]
                loss, kld_loss, recon_loss = loss_fn(x, recon_x, mean, log_var, multiplier*beta, latent_size=latent_size)  # Fixed: remove .squeeze() since x is already squeezed

                loss.backward()

                # Track gradients

                gradient_norm = []
                for param in vae.parameters():
                    if param.grad is not None:
                        gradient = param.grad.detach().cpu().numpy()
                        norm = np.linalg.norm(gradient)
                        gradient_norm.append(norm)
                    else:
                        gradient_norm.append(None)
                gradient_norms_iter.append(gradient_norm)


                optimizer.step()
                if optimizer_scheduler:
                    scheduler.step()

                # Test tracking
                num_batches_test = 0
                for i_test, (x_test, x_test_noisy) in enumerate(test_loader):
                    num_batches_test += 1
                    x_test = x_test.squeeze().to(device)  # Fixed: move x_test to device
                    x_test_noisy = x_test_noisy.squeeze().to(device)
                    
                    if x_test_noisy.ndim == 1:
                        x_test_noisy = x_test_noisy.unsqueeze(0)
                        
                    if x_test.ndim == 1:
                        x_test = x_test.unsqueeze(0)
                        
                    with torch.no_grad():
                        if model == 'CNN':
                            recon_x_test, mean_test, log_var_test = vae(x_test_noisy.float(), architecture=architecture)
                        elif model == 'FC':
                            recon_x_test, mean_test, log_var_test = vae(x_test_noisy.float())
                        elif model == 'CNNres':
                            recon_x_test, mean_test, log_var_test = vae(x_test_noisy.float())

                        _, _, recon_loss_test = loss_fn(x_test, recon_x_test,
                                                        mean_test, log_var_test, beta)


                        running_bce_test += recon_loss_test.item()

                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_kld_beta += multiplier * kld_loss.item()
                running_bce += recon_loss.item()

                num_batches_train += 1
                iter_count += 1

                beta_scheduled_track.append(multiplier*beta)
                kld_scheduled_track.append(multiplier * kld_loss.item())

            elbo_w_neg.append(-running_loss / num_batches_train)
            kl_div.append(running_kld / num_batches_train)
            kl_div_w.append(beta * running_kld / num_batches_train)
            kl_beta_scheduled_track.append(running_kld_beta / num_batches_train)
            loss_train.append(running_bce / num_batches_train)
            loss_test.append(running_bce_test / num_batches_train)
            
            average_gradient_norms_epoch.append(np.mean(gradient_norms_iter, axis=0).tolist())

            if not cluster:
                wandb.log({"-ELBO_w": -running_loss / num_batches_train,
                           "KL-D": running_kld / num_batches_train,
                           #"BCE": running_bce / num_batches,
                           #"BCE_test": recon_loss_test / num_batches,
                           #"BCE_weighted": 0.01 * running_bce / num_batches,
                           "MSE": running_bce / num_batches_train,
                           "MSE_test": recon_loss_test / num_batches_train,
                           "KL-D weighted": beta * running_kld / num_batches_train})

            logger.info(f"Epoch: {epoch}, "
                        f"LR: {optimizer.param_groups[0]['lr']}, "
                        f"beta-sch: {multiplier*beta}, "
                        f"-ELBO: {-running_loss / num_batches_train}, "
                        f"KL-D: {running_kld / num_batches_train}, "
                        f"BCE: {running_bce / num_batches_train}"
                        )
        logger.info('Finished Training')

        # Save the model with fold-specific filename
        model_checkpoint_path = os.path.join(config.experiment_dir, f"VAE_{model}_beta_{beta}_rs_{random_seed}_fold{s_number}.pth")
        torch.save({
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'loss': loss_train[-1] if loss_train else 0,
            'config': {
                'model': model,
                'architecture': architecture,
                'beta': beta,
                'latent_size': latent_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'noise_type': noise_type,
                'random_seed': random_seed,
                'fold_number': s_number
            }
        }, model_checkpoint_path)
        logger.info(f'Model checkpoint saved for fold {s_number}: {model_checkpoint_path}')

        # Calculate final comprehensive metrics after training is complete
        logger.info(f'Computing final comprehensive metrics for fold {s_number}...')
        
        # Set model to evaluation mode
        vae.eval()
        with torch.no_grad():
            # Get full train dataset reconstruction
            train_full = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False)
            for x_train_batch, x_train_noisy_batch in train_full:
                x_train_batch = x_train_batch.squeeze().to(device)
                x_train_noisy_batch = x_train_noisy_batch.squeeze().to(device)
                
                if x_train_noisy_batch.ndim == 1:
                    x_train_noisy_batch = x_train_noisy_batch.unsqueeze(0)
                if x_train_batch.ndim == 1:
                    x_train_batch = x_train_batch.unsqueeze(0)

                if model == 'CNN':
                    recon_x_train_batch, _, _ = vae(x_train_noisy_batch.float(), architecture=architecture)
                elif model == 'FC':
                    recon_x_train_batch, _, _ = vae(x_train_noisy_batch.float())
                elif model == 'CNNres':
                    recon_x_train_batch, _, _ = vae(x_train_noisy_batch.float())
                
                # Convert to numpy for metrics calculation
                x_train_np = x_train_batch.cpu().float().numpy()
                recon_x_train_np = recon_x_train_batch.cpu().float().numpy()
                
                # Compute comprehensive training metrics
                final_pcorr_train = mapf.compute_pearson_corr(x_train_np, recon_x_train_np, mode='total')
                final_rmse_train = mapf.compute_rmse(x_train_np, recon_x_train_np, mode='total') 
                final_psnr_train = mapf.compute_psnr(x_train_np, recon_x_train_np, mode='total')
                final_mse_train = mapf.compute_mse(x_train_np, recon_x_train_np, mode='total')
                final_spearman_train = mapf.compute_spearman_corr(x_train_np, recon_x_train_np, mode='total')
                final_snr_train = mapf.compute_snr(x_train_np, recon_x_train_np, mode='total')
                # DTW and LSD computation enabled for comprehensive evaluation
                final_dtw_train = mapf.compute_dtw(x_train_np, recon_x_train_np, mode='total')
                final_lsd_train = mapf.compute_lsd(x_train_np, recon_x_train_np, mode='total')
                final_nmae_range_train = mapf.compute_nmae(x_train_np, recon_x_train_np, norm='range', mode='total')
                final_nmae_l1_train = mapf.compute_nmae(x_train_np, recon_x_train_np, norm='l1', mode='total')
                final_nmae_mean_train = mapf.compute_nmae(x_train_np, recon_x_train_np, norm='mean', mode='total')
                
                # Add single final values to lists (using actual DTW and LSD values)
                # Convert all NumPy values to Python native types for consistent serialization
                pcorr_train = [float(final_pcorr_train)]
                rmse_train = [float(final_rmse_train)]
                psnr_train = [float(final_psnr_train)]
                mse_train = [float(final_mse_train)]
                spearman_train = [float(final_spearman_train)]
                snr_train = [float(final_snr_train)]
                dtw_train = [float(final_dtw_train)]  # Using actual DTW computation
                lsd_train = [float(final_lsd_train)]  # Using actual LSD computation
                nmae_range_train = [float(final_nmae_range_train)]
                nmae_l1_train = [float(final_nmae_l1_train)]
                nmae_mean_train = [float(final_nmae_mean_train)]
                
            # Get full test dataset reconstruction
            test_full = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)
            for x_test_batch, x_test_noisy_batch in test_full:
                x_test_batch = x_test_batch.squeeze().to(device)
                x_test_noisy_batch = x_test_noisy_batch.squeeze().to(device)
                
                if x_test_noisy_batch.ndim == 1:
                    x_test_noisy_batch = x_test_noisy_batch.unsqueeze(0)
                if x_test_batch.ndim == 1:
                    x_test_batch = x_test_batch.unsqueeze(0)

                if model == 'CNN':
                    recon_x_test_batch, _, _ = vae(x_test_noisy_batch.float(), architecture=architecture)
                elif model == 'FC':
                    recon_x_test_batch, _, _ = vae(x_test_noisy_batch.float())
                elif model == 'CNNres':
                    recon_x_test_batch, _, _ = vae(x_test_noisy_batch.float())
                
                # Convert to numpy for metrics calculation
                x_test_np = x_test_batch.cpu().float().numpy()
                recon_x_test_np = recon_x_test_batch.cpu().float().numpy()
                
                # Compute comprehensive test metrics
                final_pcorr_test = mapf.compute_pearson_corr(x_test_np, recon_x_test_np, mode='total')
                final_rmse_test = mapf.compute_rmse(x_test_np, recon_x_test_np, mode='total')
                final_psnr_test = mapf.compute_psnr(x_test_np, recon_x_test_np, mode='total')
                final_mse_test = mapf.compute_mse(x_test_np, recon_x_test_np, mode='total')
                final_spearman_test = mapf.compute_spearman_corr(x_test_np, recon_x_test_np, mode='total')
                final_snr_test = mapf.compute_snr(x_test_np, recon_x_test_np, mode='total')
                # DTW and LSD computation enabled for comprehensive evaluation
                final_dtw_test = mapf.compute_dtw(x_test_np, recon_x_test_np, mode='total')
                final_lsd_test = mapf.compute_lsd(x_test_np, recon_x_test_np, mode='total')
                final_nmae_range_test = mapf.compute_nmae(x_test_np, recon_x_test_np, norm='range', mode='total')
                final_nmae_l1_test = mapf.compute_nmae(x_test_np, recon_x_test_np, norm='l1', mode='total')
                final_nmae_mean_test = mapf.compute_nmae(x_test_np, recon_x_test_np, norm='mean', mode='total')
                
                # Add single final values to lists (using actual DTW and LSD values)
                # Convert all NumPy values to Python native types for consistent serialization
                pcorr_test = [float(final_pcorr_test)]
                rmse_test = [float(final_rmse_test)]
                psnr_test = [float(final_psnr_test)]
                mse_test = [float(final_mse_test)]
                spearman_test = [float(final_spearman_test)]
                snr_test = [float(final_snr_test)]
                dtw_test = [float(final_dtw_test)]  # Using actual DTW computation
                lsd_test = [float(final_lsd_test)]  # Using actual LSD computation
                nmae_range_test = [float(final_nmae_range_test)]
                nmae_l1_test = [float(final_nmae_l1_test)]
                nmae_mean_test = [float(final_nmae_mean_test)]
                
            # Hidden Test Set Evaluation
            if hidden_loader is not None:
                 logger.info(f'Computing metrics for hidden test set...')
                 for x_hidden_batch, x_hidden_noisy_batch in hidden_loader:
                     x_hidden_batch = x_hidden_batch.squeeze().to(device)
                     x_hidden_noisy_batch = x_hidden_noisy_batch.squeeze().to(device)
                     
                     if x_hidden_noisy_batch.ndim == 1:
                        x_hidden_noisy_batch = x_hidden_noisy_batch.unsqueeze(0)
                     if x_hidden_batch.ndim == 1:
                        x_hidden_batch = x_hidden_batch.unsqueeze(0)

                     if model == 'CNN':
                         recon_x_hidden_batch, _, _ = vae(x_hidden_noisy_batch.float(), architecture=architecture)
                     elif model == 'FC':
                         recon_x_hidden_batch, _, _ = vae(x_hidden_noisy_batch.float())
                     elif model == 'CNNres':
                         recon_x_hidden_batch, _, _ = vae(x_hidden_noisy_batch.float())
                     
                     # Convert to numpy
                     x_hidden_np = x_hidden_batch.cpu().float().numpy()
                     recon_x_hidden_np = recon_x_hidden_batch.cpu().float().numpy()
                     
                     # Compute hidden metrics
                     final_pcorr_hidden = mapf.compute_pearson_corr(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_rmse_hidden = mapf.compute_rmse(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_psnr_hidden = mapf.compute_psnr(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_mse_hidden = mapf.compute_mse(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_spearman_hidden = mapf.compute_spearman_corr(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_snr_hidden = mapf.compute_snr(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_dtw_hidden = mapf.compute_dtw(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_lsd_hidden = mapf.compute_lsd(x_hidden_np, recon_x_hidden_np, mode='total')
                     final_nmae_range_hidden = mapf.compute_nmae(x_hidden_np, recon_x_hidden_np, norm='range', mode='total')
                     final_nmae_l1_hidden = mapf.compute_nmae(x_hidden_np, recon_x_hidden_np, norm='l1', mode='total')
                     final_nmae_mean_hidden = mapf.compute_nmae(x_hidden_np, recon_x_hidden_np, norm='mean', mode='total')
            else:
                final_pcorr_hidden = final_rmse_hidden = final_psnr_hidden = final_mse_hidden = final_spearman_hidden = final_snr_hidden = final_dtw_hidden = final_lsd_hidden = final_nmae_range_hidden = final_nmae_l1_hidden = final_nmae_mean_hidden = np.nan

        logger.info(f'Final metrics computed for fold {s_number}')
        logger.info(f'Train - PCC: {final_pcorr_train:.4f}, RMSE: {final_rmse_train:.4f}, PSNR: {final_psnr_train:.4f}')
        logger.info(f'Test - PCC: {final_pcorr_test:.4f}, RMSE: {final_rmse_test:.4f}, PSNR: {final_psnr_test:.4f}')

        # Save denoised signals for future analysis (in try-except to prevent training failure)
        try:
            logger.info(f'Saving denoised signals for fold {s_number}...')
            
            # Create subdirectory for denoised signals
            denoised_dir = os.path.join(config.experiment_dir, "denoised_signals")
            if not os.path.exists(denoised_dir):
                os.makedirs(denoised_dir)
            
            # Save training data and denoised outputs
            train_signals_dict = {
                'original_clean': x_train_np,          # Original clean training signals  
                'noisy_input': x_train_noisy_batch.cpu().float().numpy(),  # Noisy input to the model
                'denoised_output': recon_x_train_np,   # Model's denoised output
                'fold_number': s_number,
                'data_type': 'train'
            }
            
            train_file_path = os.path.join(denoised_dir, f"fold{s_number}_train_signals.npz")
            np.savez_compressed(train_file_path, **train_signals_dict)
            logger.info(f'✅ Training signals saved: {train_file_path}')
            
            # Save test data and denoised outputs  
            test_signals_dict = {
                'original_clean': x_test_np,           # Original clean test signals
                'noisy_input': x_test_noisy_batch.cpu().float().numpy(),   # Noisy input to the model
                'denoised_output': recon_x_test_np,    # Model's denoised output
                'fold_number': s_number,
                'data_type': 'test'
            }
            
            test_file_path = os.path.join(denoised_dir, f"fold{s_number}_test_signals.npz")
            np.savez_compressed(test_file_path, **test_signals_dict)
            logger.info(f'✅ Test signals saved: {test_file_path}')

            # Save Hidden Test Set signals
            if hidden_loader is not None and 'x_hidden_np' in locals():
                try:
                    hidden_signals_dict = {
                        'original_clean': x_hidden_np,
                        'noisy_input': x_hidden_noisy_batch.cpu().float().numpy(),
                        'denoised_output': recon_x_hidden_np,
                        'fold_number': s_number,
                        'data_type': 'hidden_test'
                    }
                    hidden_file_path = os.path.join(denoised_dir, f"fold{s_number}_hidden_test_signals.npz")
                    np.savez_compressed(hidden_file_path, **hidden_signals_dict)
                    logger.info(f'✅ Hidden Test signals saved: {hidden_file_path}')
                except Exception as e:
                    logger.warning(f'⚠️ Failed to save hidden test signals: {e}')

            # Save metadata for easy reference
            metadata = {
                'fold_number': s_number,
                'model_type': model,
                'architecture': architecture,
                'beta': beta,
                'latent_size': latent_size,
                'noise_type': noise_type,
                'num_epochs': num_epochs,
                'train_samples': int(x_train_np.shape[0]),
                'test_samples': int(x_test_np.shape[0]),
                'signal_length': int(x_train_np.shape[1]),
                'final_metrics': {
                    # Core metrics (originally saved)
                    'train_pcc': float(final_pcorr_train),
                    'train_rmse': float(final_rmse_train),
                    'train_psnr': float(final_psnr_train),
                    'test_pcc': float(final_pcorr_test),
                    'test_rmse': float(final_rmse_test),
                    'test_psnr': float(final_psnr_test),
                    # Additional comprehensive metrics
                    'train_mse': float(final_mse_train),
                    'train_spearman': float(final_spearman_train),
                    'train_snr': float(final_snr_train),
                    'train_nmae_range': float(final_nmae_range_train),
                    'train_nmae_l1': float(final_nmae_l1_train),
                    'train_nmae_mean': float(final_nmae_mean_train),
                    'test_mse': float(final_mse_test),
                    'test_spearman': float(final_spearman_test),
                    'test_snr': float(final_snr_test),
                    'test_nmae_range': float(final_nmae_range_test),
                    'test_nmae_l1': float(final_nmae_l1_test),
                    'test_nmae_mean': float(final_nmae_mean_test),
                    # DTW and LSD metrics (using computed values)
                    'train_dtw': float(final_dtw_train),
                    'train_lsd': float(final_lsd_train),
                    'test_dtw': float(final_dtw_test),
                    'test_lsd': float(final_lsd_test),
                    # Hidden Test metrics
                    'hidden_pcc': float(final_pcorr_hidden),
                    'hidden_rmse': float(final_rmse_hidden),
                    'hidden_psnr': float(final_psnr_hidden),
                    'hidden_mse': float(final_mse_hidden),
                    'hidden_spearman': float(final_spearman_hidden),
                    'hidden_snr': float(final_snr_hidden),
                    'hidden_nmae_range': float(final_nmae_range_hidden),
                    'hidden_nmae_l1': float(final_nmae_l1_hidden),
                    'hidden_nmae_mean': float(final_nmae_mean_hidden),
                    'hidden_dtw': float(final_dtw_hidden),
                    'hidden_lsd': float(final_lsd_hidden)
                }
            }
            
            metadata_file_path = os.path.join(denoised_dir, f"fold{s_number}_metadata.json")
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f'✅ Metadata saved: {metadata_file_path}')
            
            logger.info(f'✅ All denoised signals successfully saved for fold {s_number}')
            
        except Exception as e:
            logger.warning(f'⚠️ Failed to save denoised signals for fold {s_number}: {str(e)}')
            logger.warning('Training will continue despite saving failure...')

        summary_dict = {
            'architecture': model,
            'architecture_type': architecture,
            'loss_type': 'mse',
            'n_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'latent_size': latent_size,
            'beta': beta,
            'noise_type': noise_type,
            'elbo_neg': elbo_w_neg,
            'kl_div': kl_div,
            'kl_div_w': kl_div_w,
            'loss_train': loss_train,
            'loss_test': loss_test,
            'pcorr_train': pcorr_train,
            'pcorr_test': pcorr_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'psnr_train': psnr_train,
            'psnr_test': psnr_test,
            # Added comprehensive metrics to summary_dict
            'mse_train': mse_train,  # Added
            'mse_test': mse_test,  # Added
            'spearman_train': spearman_train,  # Added
            'spearman_test': spearman_test,  # Added
            'snr_train': snr_train,  # Added
            'snr_test': snr_test,  # Added
            'dtw_train': dtw_train,  # Added
            'dtw_test': dtw_test,  # Added
            'lsd_train': lsd_train,  # Added
            'lsd_test': lsd_test,  # Added
            'nmae_range_train': nmae_range_train,  # Added
            'nmae_range_test': nmae_range_test,  # Added
            'nmae_l1_train': nmae_l1_train,  # Added
            'nmae_l1_test': nmae_l1_test,  # Added
            'nmae_mean_train': nmae_mean_train,  # Added
            'nmae_mean_test': nmae_mean_test,  # Added
            # Hidden Test metrics
            'pcorr_hidden': [float(final_pcorr_hidden)],
            'rmse_hidden': [float(final_rmse_hidden)],
            'psnr_hidden': [float(final_psnr_hidden)],
            'mse_hidden': [float(final_mse_hidden)],
            'spearman_hidden': [float(final_spearman_hidden)],
            'snr_hidden': [float(final_snr_hidden)],
            'dtw_hidden': [float(final_dtw_hidden)],
            'lsd_hidden': [float(final_lsd_hidden)],
            'nmae_range_hidden': [float(final_nmae_range_hidden)],
            'nmae_l1_hidden': [float(final_nmae_l1_hidden)],
            'nmae_mean_hidden': [float(final_nmae_mean_hidden)],
            'kl_scheduled': kl_beta_scheduled_track,
            'beta_scheduled': beta_scheduled_track,
            'gradient_norms': average_gradient_norms_epoch,
            'weight_constant': w_constant,
            'random_seed': random_seed
        }
        summary_dict_list.append(summary_dict)

        # Create a dictionary with the last value of each list
        last_values = {
            'Split': f"split{s_number}",
            'elbo_neg': summary_dict['elbo_neg'][-1],
            'kl_div_w': summary_dict['kl_div_w'][-1],
            'loss_train': summary_dict['loss_train'][-1],
            'loss_test': summary_dict['loss_test'][-1],
            'pcorr_train': summary_dict['pcorr_train'][-1],
            'pcorr_test': summary_dict['pcorr_test'][-1],
            'psnr_train': summary_dict['psnr_train'][-1],
            'psnr_test': summary_dict['psnr_test'][-1],
            'rmse_train': summary_dict['rmse_train'][-1],
            'rmse_test': summary_dict['rmse_test'][-1],
            # Added comprehensive metrics to last_values
            'mse_train': summary_dict['mse_train'][-1],  # Added
            'mse_test': summary_dict['mse_test'][-1],  # Added
            'spearman_train': summary_dict['spearman_train'][-1],  # Added
            'spearman_test': summary_dict['spearman_test'][-1],  # Added
            'snr_train': summary_dict['snr_train'][-1],  # Added
            'snr_test': summary_dict['snr_test'][-1],  # Added
            'dtw_train': summary_dict['dtw_train'][-1],  # Added
            'dtw_test': summary_dict['dtw_test'][-1],  # Added
            'lsd_train': summary_dict['lsd_train'][-1],  # Added
            'lsd_test': summary_dict['lsd_test'][-1],  # Added
            'nmae_range_train': summary_dict['nmae_range_train'][-1],  # Added
            'nmae_range_test': summary_dict['nmae_range_test'][-1],  # Added
            'nmae_l1_train': summary_dict['nmae_l1_train'][-1],  # Added
            'nmae_l1_test': summary_dict['nmae_l1_test'][-1],  # Added
            'nmae_mean_train': summary_dict['nmae_mean_train'][-1],  # Added
            'nmae_mean_test': summary_dict['nmae_mean_test'][-1],  # Added
            # Hidden metrics
            'pcorr_hidden': summary_dict['pcorr_hidden'][-1],
            'rmse_hidden': summary_dict['rmse_hidden'][-1],
            'psnr_hidden': summary_dict['psnr_hidden'][-1],
            'mse_hidden': summary_dict['mse_hidden'][-1],
            'spearman_hidden': summary_dict['spearman_hidden'][-1],
            'snr_hidden': summary_dict['snr_hidden'][-1],
            'dtw_hidden': summary_dict['dtw_hidden'][-1],
            'lsd_hidden': summary_dict['lsd_hidden'][-1],
            'nmae_range_hidden': summary_dict['nmae_range_hidden'][-1],
            'nmae_l1_hidden': summary_dict['nmae_l1_hidden'][-1],
            'nmae_mean_hidden': summary_dict['nmae_mean_hidden'][-1]
        }
        # Create a DataFrame from the dictionary
        last_values_df = pd.DataFrame([last_values])
        # Append the DataFrame to the list
        summary_values.append(last_values_df)

    # Concatenate the DataFrames in the list vertically
    combined_df = pd.concat(summary_values, ignore_index=True)

    numeric_cols = combined_df.select_dtypes(include=np.number)
    average_row = numeric_cols.mean()
    std_dev_row = numeric_cols.std()

    # Add labels to the split column for average and standard deviation rows
    average_row['Split'] = 'average'
    std_dev_row['Split'] = 'st. dev.'

    average_df = pd.DataFrame(average_row).transpose()
    average_df['Split'] = 'average'
    std_dev_df = pd.DataFrame(std_dev_row).transpose()
    std_dev_df['Split'] = 'st. dev.'

    # Append the average and standard deviation rows to the dataframe
    combined_df = pd.concat([combined_df, average_df, std_dev_df], ignore_index=True)
    #combined_df = combined_df.append(average_row, ignore_index=True)
    #combined_df = combined_df.append(std_dev_row, ignore_index=True)

    # Format numbers to have exactly 5 decimal places
    combined_df = combined_df.round(5)

    # Define the path to save the Excel file
    output_path = os.path.join(config.experiment_dir, "summary_values.xlsx")
    # Save the combined DataFrame as an Excel file
    combined_df.to_excel(output_path, index=False)
    logger.info(f"Excel file 'summary_values.xlsx' saved with data")
    
    # Final summary for all folds
    logger.info("="*60)
    logger.info("TRAINING COMPLETE - ALL FOLDS SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successfully trained {n_folds} folds")
    logger.info(f"✅ All {n_folds} model checkpoints saved in: {config.experiment_dir}")
    
    # List all saved checkpoints
    for fold_idx in range(n_folds):
        checkpoint_name = f"VAE_{model}_beta_{beta}_rs_{random_seed}_fold{fold_idx}.pth"
        checkpoint_path = os.path.join(config.experiment_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            logger.info(f"✅ Fold {fold_idx}: {checkpoint_name}")
        else:
            logger.warning(f"❌ Fold {fold_idx}: {checkpoint_name} - FILE MISSING!")
    
    logger.info(f"✅ Final metrics summary saved: {output_path}")
    
    # Summary of saved denoised signals
    denoised_dir = os.path.join(config.experiment_dir, "denoised_signals")
    if os.path.exists(denoised_dir):
        logger.info(f"✅ Denoised signals directory: {denoised_dir}")
        signal_files = [f for f in os.listdir(denoised_dir) if f.endswith('.npz')]
        metadata_files = [f for f in os.listdir(denoised_dir) if f.endswith('.json')]
        logger.info(f"✅ {len(signal_files)} signal files saved (train+test for {len(signal_files)//2} folds)")
        logger.info(f"✅ {len(metadata_files)} metadata files saved")
        
        # List all saved files
        for fold_idx in range(n_folds):
            train_signal_file = f"fold{fold_idx}_train_signals.npz"
            test_signal_file = f"fold{fold_idx}_test_signals.npz" # Corresponds to VAL
            hidden_signal_file = f"fold{fold_idx}_hidden_test_signals.npz" # Corresponds to TEST (Hidden)
            metadata_file = f"fold{fold_idx}_metadata.json"
            
            if train_signal_file in signal_files:
                logger.info(f"✅ Fold {fold_idx} TRAIN signals: {train_signal_file}")
            else:
                logger.warning(f"❌ Fold {fold_idx} TRAIN signals: MISSING")
                
            if test_signal_file in signal_files:
                logger.info(f"✅ Fold {fold_idx} VAL (Fold Test) signals: {test_signal_file}")
            else:
                logger.warning(f"❌ Fold {fold_idx} VAL (Fold Test) signals: MISSING")
                
            if hidden_signal_file in signal_files:
                logger.info(f"✅ Fold {fold_idx} TEST (Hidden) signals: {hidden_signal_file}")
            # Hidden is optional, so no warning if missing unless strictly expected

            if metadata_file in metadata_files:
                logger.info(f"✅ Fold {fold_idx} metadata: {metadata_file}")
            else:
                logger.warning(f"❌ Fold {fold_idx} metadata: MISSING")
    else:
        logger.warning("❌ Denoised signals directory not found - signals may not have been saved")
    
    logger.info("="*60)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"\n🎉 Training completed! All {n_folds} folds processed successfully.")
    print(f"📁 Results saved in: {config.experiment_dir}")
    print(f"📊 Excel summary: {os.path.basename(output_path)}")
    print(f"🧠 Model checkpoints: VAE_*_fold*.pth")
    print(f"🔧 Denoised signals: denoised_signals/ directory")
    print(f"⏱️  Total time: {execution_time/60:.2f} minutes")

_CONFIG = config_flags.DEFINE_config_file("config", lock_config=False)
if __name__ == "__main__":
    app.run(main)




