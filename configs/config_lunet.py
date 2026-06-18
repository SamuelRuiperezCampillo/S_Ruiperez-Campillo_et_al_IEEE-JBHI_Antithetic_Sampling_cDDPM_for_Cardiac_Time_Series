"""LUNet baseline configuration (`ml_collections` ConfigDict).

Faithful migration of `MAP_VAE/config_lunet.py`. Consumed at runtime by the absl
`config_flags.DEFINE_config_file('config', ...)` mechanism (passed as
`--config=configs/config_lunet.py`), not imported. The only non-mechanical change
relative to the original is that the hard-coded `/cluster/...` `working_dir`
default is now routed through `cardiac_map_diffusion.paths.experiments_root()`
so no cluster-specific absolute path is baked into the repository; override at
runtime with `--config.working_dir=...` exactly as the submission scripts do.
"""

import ml_collections
from cardiac_map_diffusion import paths

def get_config():
    config = ml_collections.ConfigDict()
    
    # Model architecture
    config.model_name = 'LUNet'
    config.architecture = 'UNet_skip_connections'
    
    # Training parameters
    config.exclude_patients_file = '' # File to exclude patients from training (e.g. for hidden test set)
    config.learning_rate = 5e-5  # Same as DAE for fair comparison
    config.num_epochs = 1000  # Updated to reasonable baseline value
    config.batch_size = 32
    config.optimizer = 'adam'
    config.optimizer_scheduler = False  # Disabled for baseline simplicity
    
    # Adam optimizer parameters (same as DAE)
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.99
    
    # Early stopping
    config.early_stopping = True
    config.early_stopping_patience = 10
    
    # Data parameters
    config.noise_type = 'allmixed'  # Same as DAE and VAE for fair comparison
    config.num_folds = 4
    config.seed = 17  # Same as DAE and VAE
    config.seed_split = 29  # Same as DAE (k-fold splitting seed)
    config.split_number = 0  # Default split number (overridden by submission script)
    
    # System parameters
    config.cluster = True  # Set to False for local development
    config.num_workers = 4
    config.prefetch_factor = 2
    
    # Experiment settings
    config.experiment_name = f'LUNet_baseline_{config.noise_type}_es{config.early_stopping_patience}'
    config.working_dir = str(paths.experiments_root())  # Match DAE path
    
    return config