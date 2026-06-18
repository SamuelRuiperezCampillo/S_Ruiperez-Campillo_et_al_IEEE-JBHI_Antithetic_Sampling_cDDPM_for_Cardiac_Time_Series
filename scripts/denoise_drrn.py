"""Train the DRRN denoising baseline across all k-folds (entry script).

Faithful migration of ``MAP_VAE/drrn_denoising_allfolds.py`` (the root copy that
the ``submit_jobs_drrn.sh`` launcher invokes). Only mechanical edits were applied:
imports were rewritten to the ``cardiac_map_diffusion`` package layout and this
module docstring was added. All training/evaluation logic is byte-for-byte
unchanged. Run via the absl config-file mechanism, e.g.
``python scripts/denoise_drrn.py --config=configs/config_drrn.py``.
"""
import torch
import torch.nn.functional as F
from cardiac_map_diffusion.data.retrieve_dataset import retrieveDataSet
from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data, get_train_test_kfolds
import os
import logging
import json
import numpy as np
import cardiac_map_diffusion.metrics.map_functions_baselines as mapf
import wandb
from absl import app
from ml_collections import config_flags
import time
import pandas as pd
from cardiac_map_diffusion.models.drrn_model import DRRNModel

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
    noise_type = config.noise_type
    working_dir = config.working_dir
    cluster = config.cluster
    optimizer_type = config.optimizer
    n_folds = config.num_folds
    n_workers = config.num_workers
    pref_factor = config.prefetch_factor
    hidden_size = getattr(config, 'hidden_size', 64)  # LSTM hidden size
    
    # Early stopping parameters
    early_stopping = getattr(config, 'early_stopping', False)
    early_stopping_patience = getattr(config, 'early_stopping_patience', 10)
    
    # Adam parameters
    adam_beta1 = getattr(config, 'adam_beta1', 0.9)
    adam_beta2 = getattr(config, 'adam_beta2', 0.99)

    for s_number in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Starting DRRN training for FOLD {s_number + 1}/{n_folds}")
        print(f"{'='*60}")
        
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
        
        # Split hidden test set if configuration is provided
        # -------------------------------------------------------------
        # Identify excluded patients for hidden test set
        excluded_patients = []
        if hasattr(config, 'exclude_patients_file') and config.exclude_patients_file:
            exclusion_file_path = config.exclude_patients_file
            # Handle relative paths
            if not os.path.isabs(exclusion_file_path):
                # Try to find it relative to current working directory or project root
                possible_paths = [
                    os.path.join(os.getcwd(), exclusion_file_path),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), exclusion_file_path),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), exclusion_file_path)
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        exclusion_file_path = p
                        break
            
            if os.path.exists(exclusion_file_path):
                try:
                    exclusion_df = pd.read_csv(exclusion_file_path)
                    # Check if 'pat_ID' column exists
                    if 'pat_ID' in exclusion_df.columns:
                        # Ensure pat_ID is treated as string for comparison
                        excluded_patients = exclusion_df['pat_ID'].astype(str).tolist()
                        logger.info(f"Loaded {len(excluded_patients)} patients to exclude from {exclusion_file_path}")
                    else:
                        logger.warning(f"Exclusion file {exclusion_file_path} does not have 'pat_ID' column. Ignoring.")
                except Exception as e:
                    logger.warning(f"Failed to read exclusion file: {e}")
            else:
                 logger.warning(f"Exclusion file not found at: {exclusion_file_path}")

        # Remove excluded patients from the complete dataset and create hidden set
        df_hidden = pd.DataFrame() # Empty placeholder
        
        if excluded_patients:
            # Create hidden test set
            df_hidden = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(excluded_patients)].copy()
            
            # Remove from main dataset
            original_len = len(MAP_vent_complete)
            MAP_vent_complete = MAP_vent_complete[~MAP_vent_complete['pat_ID'].isin(excluded_patients)].reset_index(drop=True)
            new_len = len(MAP_vent_complete)
            
            logger.info(f"Split dataset: {len(df_hidden)} samples moved to hidden test set.")
            logger.info(f"Training/Validation set reduced from {original_len} to {new_len} samples.")
            
            if len(df_hidden) == 0:
                logger.warning(f"WARNING: No samples found for excluded patients: {excluded_patients}")
        # -------------------------------------------------------------

        noise_params = mapf.find_noise_params(noise_type)
        if noise_type == 'ep' or noise_type == 'allmixed':
            arrays = mapf.get_np_noisearrays(MAP_vent_complete)
        else:
            arrays = []

        if not cluster:
            wandb.init(
                # set the wandb project where this run will be logged
                project="DRRN_pytorch_baseline",
                name=f"model_DRRN_fold_{s_number}_ntype_{noise_type[:3]}_es{early_stopping_patience if early_stopping else 'off'}",
                # track hyperparameters and run metadata
                config={
                    "learning_rate": learning_rate,
                    "adam_beta1": adam_beta1,
                    "adam_beta2": adam_beta2,
                    "architecture": "DRRN_LSTM",
                    "dataset": "ventricular_MAP",
                    "max_epochs": num_epochs,
                    "hidden_size": hidden_size,
                    "early_stopping": early_stopping,
                    "early_stopping_patience": early_stopping_patience if early_stopping else None
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

        input_dim = X_std_train.shape[-1]
        drrn = DRRNModel(input_dim=input_dim, hidden_size=hidden_size, adaptive=False).float()
        drrn = drrn.to(device)

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(drrn.parameters(), lr=learning_rate, 
                                       betas=(adam_beta1, adam_beta2))
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(drrn.parameters(), lr=learning_rate)

        # No scheduler for baseline simplicity (as fixed in DAE and LUNet)

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

        average_gradient_norms_epoch = []

        # Early stopping variables
        best_test_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}", end=" - ")

            running_loss = 0.0
            running_loss_test = 0.0
            num_batches_train = 0
            gradient_norms_iter = []

            for i, (x, x_noisy) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.squeeze().float().to(device)  # Clean target signal
                x_noisy = x_noisy.squeeze().float().to(device)  # Noisy input signal
                
                # Reshape for DRRN: Expected (Batch, Length, 1) or (Batch, Length) which internal forward handles
                # If x_noisy became 1D (Length) due to squeeze on single batch, we need to fix it
                if x_noisy.ndim == 1:
                     x_noisy = x_noisy.unsqueeze(0) # (1, Length)
                
                # Check target shape matching
                if x.ndim == 1:
                     x = x.unsqueeze(0)
                
                # Forward pass through DRRN: input noisy, target clean
                x_recon = drrn(x_noisy)  # Reconstruct from noisy input
                
                # Compute loss between reconstruction and clean target
                loss = torch.nn.functional.mse_loss(x_recon, x, reduction='mean')

                loss.backward()

                # Track gradients
                gradient_norm = []
                for param in drrn.parameters():
                    if param.grad is not None:
                        gradient = param.grad.detach().cpu().numpy()
                        norm = np.linalg.norm(gradient)
                        gradient_norm.append(norm)
                    else:
                        gradient_norm.append(None)
                gradient_norms_iter.append(gradient_norm)

                optimizer.step()

                running_loss += loss.item()
                num_batches_train += 1

            # Test tracking (MOVED OUTSIDE training loop)
            num_batches_test = 0
            for i_test, (x_test, x_test_noisy) in enumerate(test_loader):
                num_batches_test += 1
                x_test = x_test.squeeze().float().to(device)  # Clean target
                x_test_noisy = x_test_noisy.squeeze().float().to(device)  # Noisy input
                
                # Reshape for DRRN: Expected (Batch, Length) which internal forward handles
                if x_test_noisy.ndim == 1:
                     x_test_noisy = x_test_noisy.unsqueeze(0)

                with torch.no_grad():
                    x_test_recon = drrn(x_test_noisy)  # Reconstruct from noisy
                    loss_test_val = torch.nn.functional.mse_loss(x_test_recon, x_test, reduction='mean')
                    running_loss_test += loss_test_val.item()

            loss_train.append(running_loss / num_batches_train)
            loss_test.append(running_loss_test / num_batches_test)
            
            # Print epoch completion with losses
            train_loss_avg = running_loss / num_batches_train
            test_loss_avg = running_loss_test / num_batches_test
            print(f"Train Loss: {train_loss_avg:.6f}, Val Loss: {test_loss_avg:.6f}")
            
            average_gradient_norms_epoch.append(np.mean(gradient_norms_iter, axis=0).tolist())

            if not cluster:
                wandb.log({
                    "MSE_train": running_loss / num_batches_train,
                    "MSE_test": running_loss_test / num_batches_test
                })

            logger.info(f"Epoch: {epoch}, "
                        f"LR: {optimizer.param_groups[0]['lr']}, "
                        f"MSE_train: {running_loss / num_batches_train}, "
                        f"MSE_test: {running_loss_test / num_batches_test}"
                        )
            
            # Early stopping check
            current_test_loss = running_loss_test / num_batches_test
            if early_stopping:
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_epoch = epoch
                    patience_counter = 0
                    print(f"    → New best model! Saved at epoch {epoch + 1}")
                    # Save best model
                    best_model_path = os.path.join(config.experiment_dir, f"DRRN_best_rs_{random_seed}_fold{s_number}.pth")
                    torch.save({
                        'model_state_dict': drrn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': current_test_loss,
                        'config': {
                            'model': 'DRRN',
                            'input_dim': input_dim,
                            'hidden_size': hidden_size,
                            'learning_rate': learning_rate,
                            'num_epochs': num_epochs,
                            'batch_size': batch_size,
                            'noise_type': noise_type,
                            'random_seed': random_seed,
                            'fold_number': s_number
                        }
                    }, best_model_path)
                    logger.info(f"New best model saved at epoch {epoch} with test loss: {current_test_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"    → Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    
                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 EARLY STOPPING triggered at epoch {epoch + 1}")
                    print(f"   Best epoch was {best_epoch + 1} with test loss: {best_test_loss:.6f}")
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with test loss: {best_test_loss:.6f}")
                    # Load best model for final evaluation
                    if os.path.exists(best_model_path):
                        checkpoint = torch.load(best_model_path)
                        drrn.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"Loaded best model from epoch {best_epoch}")
                    break
                        
        print(f"\n✅ Fold {s_number + 1} training completed!")
        logger.info('Finished Training')

        # Save the final model with fold-specific filename
        final_epoch = min(len(loss_train), num_epochs) if loss_train else 0
        model_checkpoint_path = os.path.join(config.experiment_dir, f"DRRN_rs_{random_seed}_fold{s_number}.pth")
        torch.save({
            'model_state_dict': drrn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': final_epoch,
            'final_epoch': final_epoch,
            'best_epoch': best_epoch if early_stopping else final_epoch,
            'loss': loss_train[-1] if loss_train else 0,
            'early_stopped': patience_counter >= early_stopping_patience if early_stopping else False,
            'config': {
                'model': 'DRRN',
                'input_dim': input_dim,
                'hidden_size': hidden_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'actual_epochs': final_epoch,
                'batch_size': batch_size,
                'noise_type': noise_type,
                'random_seed': random_seed,
                'fold_number': s_number,
                'early_stopping': early_stopping,
                'early_stopping_patience': early_stopping_patience
            }
        }, model_checkpoint_path)
        logger.info(f'Final model checkpoint saved for fold {s_number}: {model_checkpoint_path}')

        # Calculate final comprehensive metrics after training is complete
        logger.info(f'Computing final comprehensive metrics for fold {s_number}...')
        
        # Set model to evaluation mode
        drrn.eval()
        with torch.no_grad():
            # Get full train dataset reconstruction
            train_full = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False)
            for x_train_batch, x_train_noisy_batch in train_full:
                x_train_batch = x_train_batch.squeeze().float().to(device)  # Ensure float32 type
                x_train_noisy_batch = x_train_noisy_batch.squeeze().float().to(device)  # Ensure float32 type
                
                # Reshape for DRRN: Expected (Batch, Length) which internal forward handles
                if x_train_noisy_batch.ndim == 1:
                     x_train_noisy_batch = x_train_noisy_batch.unsqueeze(0)

                # Forward pass through DRRN
                recon_x_train_batch = drrn(x_train_noisy_batch.float())
                
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
                x_test_batch = x_test_batch.squeeze().float().to(device)  # Ensure float32 type
                x_test_noisy_batch = x_test_noisy_batch.squeeze().float().to(device)  # Ensure float32 type
                
                # Reshape for DRRN: Expected (Batch, Length) which internal forward handles
                if x_test_noisy_batch.ndim == 1:
                     x_test_noisy_batch = x_test_noisy_batch.unsqueeze(0)

                # Forward pass through DRRN
                recon_x_test_batch = drrn(x_test_noisy_batch.float())
                
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

                     recon_x_hidden_batch = drrn(x_hidden_noisy_batch.float())
                     
                     # Convert to numpy
                     x_hidden_np = x_hidden_batch.cpu().float().numpy()
                     recon_x_hidden_np = recon_x_hidden_batch.cpu().float().numpy()
                     
                     # Compute hidden metrics (using mapf functions)
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
                'model_type': 'DRRN',
                'input_dim': input_dim,
                'hidden_size': hidden_size,
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
                    # Placeholder metrics (set to 0.0 as per training logic)
                    'train_dtw': 0.0,
                    'train_lsd': 0.0,
                    'test_dtw': 0.0,
                    'test_lsd': 0.0,
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
            'architecture': 'DRRN',
            'architecture_type': 'LSTM_recurrent',
            'loss_type': 'mse',
            'n_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'noise_type': noise_type,
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
            'gradient_norms': average_gradient_norms_epoch,
            'random_seed': random_seed
        }
        summary_dict_list.append(summary_dict)

        # Create a dictionary with the last value of each list
        last_values = {
            'Split': f"split{s_number}",
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
        checkpoint_name = f"DRRN_rs_{random_seed}_fold{fold_idx}.pth"
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
            test_signal_file = f"fold{fold_idx}_test_signals.npz"
            hidden_signal_file = f"fold{fold_idx}_hidden_test_signals.npz"
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
    print(f"🧠 Model checkpoints: DRRN_*_fold*.pth")
    print(f"🔧 Denoised signals: denoised_signals/ directory")
    print(f"⏱️  Total time: {execution_time/60:.2f} minutes")

_CONFIG = config_flags.DEFINE_config_file("config", lock_config=False)
if __name__ == "__main__":
    app.run(main)