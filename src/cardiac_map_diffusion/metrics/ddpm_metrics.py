''' This script contains the metrics used to evaluate the performance of the denoising network.

    Further a function to log the metrics is provided.

    The functions are created to handle batches.

    Faithful migration of ``Diffusion_MAP_fullpipeline_final/metrics.py`` (the DDPM
    diffusion track's metrics). The body is copied verbatim; only this docstring and
    the import-annotation comment below have been added. No logic, math, thresholds
    or metric formulas have been changed -- in particular ``compute_mse`` returns the
    mean squared error and the commented-out ``rmse = torch.sqrt(mse)`` line is kept.
'''

import torch
import torch.nn as nn
import numpy as np
from cardiac_map_diffusion.data.data_diffusion import plot_and_log_signals
from cardiac_map_diffusion.data.generate_noise import introduce_several_noises
from cardiac_map_diffusion.baselines.filters import butterworth_notch

def compute_pearson_corr(signals, signals_denoised, mode='total'):
    x = signals - signals.mean(dim=-1, keepdim=True)
    y = signals_denoised - signals_denoised.mean(dim=-1, keepdim=True)
    
    covariance = (x * y).sum(dim=-1) 
    std1 = torch.sqrt((x ** 2).sum(dim=-1))
    std2 = torch.sqrt((y ** 2).sum(dim=-1))
    
    pcorr = covariance / (std1 * std2)
    
    if mode == 'individual':
        return pcorr
    if mode == 'sum':
        return pcorr.sum()
    elif mode == 'stats':
        mean = pcorr.mean()
        max = pcorr.max()
        min = pcorr.min()
        var = pcorr.var()
        return {'mean': mean, 'max': max, 'min': min, 'var': var}
    elif mode == 'total':
        return pcorr.mean()

def compute_mse(signals, signals_denoised, mode='total'):
    mse = ((signals - signals_denoised) ** 2).mean(dim=-1)
    #rmse = torch.sqrt(mse)
    
    if mode == 'individual':
        return mse
    elif mode == 'sum':
        return mse.sum()
    elif mode == 'total':
        return mse.mean()


def compute_psnr(signals, signals_denoised, mode='total'):
    mse = ((signals - signals_denoised)**2).mean(dim=-1)
    max_val = torch.max(signals.max(dim=-1)[0], signals_denoised.max(dim=-1)[0])
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    if mode == 'individual':
        return psnr
    elif mode == 'sum':
        return psnr.sum()
    elif mode == 'total':
        return psnr.mean()


def log_metrics(logger, total_loss, total_pearson_corr, total_mse,
                total_psnr, global_step, num_samples, name=""):
    if total_loss != None:
        avg_loss = total_loss / num_samples
        logger.add_scalar(f"MSE per diffusion step {name}", avg_loss, global_step=global_step)
    avg_pears_corr = total_pearson_corr / num_samples
    logger.add_scalar(f"Pearson Correlation {name}", avg_pears_corr, global_step=global_step)
    avg_mse = total_mse / num_samples
    logger.add_scalar(f"Reconstruction MSE {name}", avg_mse, global_step=global_step)
    avg_psnr = total_psnr / num_samples
    logger.add_scalar(f"PSNR {name}", avg_psnr, global_step=global_step)


def log_filtered_signal_metrics(logger, signals_original, signals_noisy, filter_fn, 
                                global_step=0, num_samples=1, name=""):
    """
    Filter the noisy signals using the provided filter function and compute metrics
    between the filtered and original signals.
    """
    # Apply the provided filter function to each noisy signal.
    filtered_signals = np.array([filter_fn(signal) for signal in signals_noisy])
    filtered_signals_tensor = torch.tensor(filtered_signals)
    signals_original_tensor = torch.tensor(signals_original)
    
    # Compute metrics between the original and the filtered signals.
    total_pearson_filtered = compute_pearson_corr(signals_original_tensor, filtered_signals_tensor, mode='sum')
    total_mse_filtered = compute_mse(signals_original_tensor, filtered_signals_tensor, mode='sum')
    total_psnr_filtered = compute_psnr(signals_original_tensor, filtered_signals_tensor, mode='sum')
    
    # Log the metrics by calling the existing log_metrics function with total_loss set to None.
    log_metrics(
        logger,
        total_loss=None,
        total_pearson_corr=total_pearson_filtered,
        total_mse=total_mse_filtered,
        total_psnr=total_psnr_filtered,
        global_step=global_step,
        num_samples=num_samples,
        name=f"Filtered {name}"
    )

    return filtered_signals


def evaluate_model(epoch, model, train_dataloader, test_dataloader, device, 
                   ep_noise_arrays, args, num_samples_train, num_samples_test,
                   signal_len, l, downstream, diffusion, 
                   logger, noise_batchwise, name=""):

    def run_evaluation(dataloader, num_samples, dataset_name=""):
        """
        Helper function to reduce verbosity (admittedly still very verbose). 
        """
        mse_loss = nn.MSELoss()
        total_loss = 0.0

        # Metrics for the denoised results
        total_pearson = 0.0
        total_mse = 0.0
        total_psnr = 0.0

        # Metrics for the noisy signals
        total_pearson_noisy = 0.0
        total_mse_noisy = 0.0
        total_psnr_noisy = 0.0

        # Pre-allocate arrays for storing signals (for downstream tasks and plotting)
        all_signals_denoised = np.zeros((num_samples, signal_len))
        all_signals_noisy = np.zeros((num_samples, signal_len))
        all_signals_original = np.zeros((num_samples, signal_len))
        curr_index = 0

        with torch.no_grad():
            for signals in dataloader:
                signals = signals.unsqueeze(1).to(device)
                
                # Preserve original clean signals
                signals_original = signals.squeeze(1)

                # Create noisy signals by introducing additional noise, samplewise for bigger variability
                x_noisy = introduce_several_noises(signals, ep_noise_arrays, noise_batchwise=False)

                # Sample random timesteps for diffusion process
                t = diffusion.sample_timesteps(signals.shape[0]).to(device)
                sqrt_alpha_hat = diffusion.alpha_hat[t].sqrt()

                # Generate noised signal x_t and compute noise residual
                x_t, noise = diffusion.noise_signal(signals, t)
                predicted_noise = model(x_t, x_noisy, sqrt_alpha_hat)
                loss = mse_loss(noise, predicted_noise)
                total_loss += loss.item() * signals.size(0)

                # Denoise the signals using 2-shot AV Monte Carlo
                signals_denoised_crude, signals_denoised_anti = diffusion.inference_antithetic(model, 1, x_noisy)
                signals_denoised = 0.5 * (signals_denoised_crude.squeeze(1) + signals_denoised_anti.squeeze(1))

                # Compute metrics using the original signals
                total_pearson += compute_pearson_corr(signals_original, signals_denoised, mode='sum')
                total_mse += compute_mse(signals_original, signals_denoised, mode='sum')
                total_psnr += compute_psnr(signals_original, signals_denoised, mode='sum')

                # Compute metrics using the original signals and the noisy signals
                total_pearson_noisy += compute_pearson_corr(signals_original, x_noisy.squeeze(1), mode='sum')
                total_mse_noisy += compute_mse(signals_original, x_noisy.squeeze(1), mode='sum')
                total_psnr_noisy += compute_psnr(signals_original, x_noisy.squeeze(1), mode='sum')

                # Store the signals for downstream tasks and plotting
                curr_batch_size = signals.size(0)
                all_signals_denoised[curr_index:(curr_index + curr_batch_size), :] = signals_denoised.detach().cpu().numpy()
                all_signals_noisy[curr_index:(curr_index + curr_batch_size), :] = x_noisy.squeeze(1).detach().cpu().numpy()
                all_signals_original[curr_index:(curr_index + curr_batch_size), :] = signals_original.detach().cpu().numpy()
                curr_index += curr_batch_size

        return (total_loss, total_pearson, total_mse, total_psnr,
                all_signals_denoised, all_signals_noisy, all_signals_original,
                total_pearson_noisy, total_mse_noisy, total_psnr_noisy)

    # Global step for logging purposes
    global_step = (epoch + 1) * l

    # Evaluate on the test dataset
    (test_loss, test_pearson, test_mse, test_psnr,
     test_signals_denoised, test_signals_noisy, test_signals_original,
     test_pearson_noisy, test_mse_noisy, test_psnr_noisy) = \
        run_evaluation(test_dataloader, num_samples_test, dataset_name="test")
        
    # Log metrics for the denoised signals (comparing original and denoised)
    log_metrics(logger, test_loss, test_pearson, test_mse, test_psnr,
                global_step, num_samples_test, name + "_test_denoised")
    # Log metrics for the noisy signals (comparing original and noisy)
    log_metrics(logger, None, test_pearson_noisy, test_mse_noisy, test_psnr_noisy,
                global_step, num_samples_test, name + "_test_noisy")
                
    # Plot and log signals for the test set (original, noisy, and denoised)
    plot_and_log_signals(logger, 
                         torch.from_numpy(test_signals_denoised).unsqueeze(1).to(device), 
                         torch.from_numpy(test_signals_noisy).unsqueeze(1).to(device), 
                         torch.from_numpy(test_signals_original).unsqueeze(1).to(device),
                         args.n_prints, global_step)

    # During training we don't track the metrics on the train set, nor do we compute the filter baseline, nor do we evaluate the APD predictions
    if train_dataloader != None:
        # Evaluate on the train dataset
        (train_loss, train_pearson, train_mse_val, train_psnr,
        train_signals_denoised, train_signals_noisy, train_signals_original,
        train_pearson_noisy, train_mse_noisy, train_psnr_noisy) = \
            run_evaluation(train_dataloader, num_samples_train, dataset_name="train")
            
        # Log metrics for the denoised signals
        log_metrics(logger, train_loss, train_pearson, train_mse_val, train_psnr,
                    global_step, num_samples_train, name + "_train_denoised")
        # Log metrics for the noisy signals (comparing original and noisy)
        log_metrics(logger, None, train_pearson_noisy, train_mse_noisy, train_psnr_noisy,
                    global_step, num_samples_train, name + "_train_noisy")

        # Filter the train and test data set and compute metrics (comparing original and filtered)
        test_signals_filtered = log_filtered_signal_metrics(logger, test_signals_original, test_signals_noisy, filter_fn=butterworth_notch, 
                            global_step=global_step, num_samples=num_samples_test, name="test")

        train_signals_filtered = log_filtered_signal_metrics(logger, train_signals_original, train_signals_noisy, filter_fn=butterworth_notch, 
                            global_step=global_step, num_samples=num_samples_train, name="train")

        # Log the APD prediction errors
        downstream.predict(train_signals_noisy, train_signals_filtered, train_signals_denoised, global_step, name='train')
        downstream.predict(test_signals_noisy, test_signals_filtered, test_signals_denoised, global_step, name='test')

