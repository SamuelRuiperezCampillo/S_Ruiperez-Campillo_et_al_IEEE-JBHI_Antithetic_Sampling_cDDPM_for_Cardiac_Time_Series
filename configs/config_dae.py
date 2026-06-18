"""DAE baseline configuration (``ml_collections`` ConfigDict).

Faithful migration of ``MAP_VAE/config_dae.py``. Consumed at runtime by the absl
``config_flags.DEFINE_config_file('config', ...)`` mechanism (passed as
``--config=configs/config_dae.py``), not imported. The only non-mechanical change
relative to the original is that the hard-coded ``/cluster/...`` ``working_dir``
default is now routed through ``cardiac_map_diffusion.paths.experiments_root()``
so no cluster-specific absolute path is baked into the repository; override at
runtime with ``--config.working_dir=...`` exactly as the submission scripts do.
"""

import os
import ml_collections
from cardiac_map_diffusion import paths

def get_config():
    config = ml_collections.ConfigDict()

    config.working_dir = str(paths.experiments_root())
    config.experiment_name = "DAE_baseline_experiment"
    config.exclude_patients_file = '' # File to exclude patients from training (e.g. for hidden test set)
    config.cluster = False
    config.seed = 17

    # DAE model (no VAE-specific parameters like beta, architecture types)
    config.optimizer = "adam" # {adam, sgd}
    config.batch_size = 32
    config.num_epochs = 10000  # Maximum epochs with early stopping
    config.learning_rate = 5e-5  # AdamOptimizer learning rate
    config.optimizer_scheduler = True  # Enable scheduler for better convergence
    
    # Adam optimizer specific parameters
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.99
    
    # Early stopping parameters
    config.early_stopping = True
    config.early_stopping_patience = 10
    config.seed = 17
    config.num_folds = 4
    config.split_number = 0
    config.num_workers = 4
    config.prefetch_factor = 2
    config.latent_size = 32  # Same as VAE for fair comparison
    config.seed_split = 29

    config.noise_type = "allmixed" # ['truncation', 'gaussian', 'spike', 'bwander', 'powerline', 'allmixed']
    return config
