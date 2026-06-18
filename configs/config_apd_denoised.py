import ml_collections

from cardiac_map_diffusion import paths

def get_config():
    config = ml_collections.ConfigDict()

    # Basic configuration
    config.working_dir = str(paths.experiments_root())
    config.experiment_name = 'apd_denoised_analysis'
    config.cluster = True
    config.random_seed = 17
    config.num_folds = 4

    # APD prediction settings
    config.label_apd = 'APD30_gs'  # Will be overridden by command line
    config.perc_minimum = 5
    config.perc_maximum = 95

    # Source of denoised signals (will be overridden by command line)
    config.denoised_signals_folder = str(paths.experiments_root())

    return config
