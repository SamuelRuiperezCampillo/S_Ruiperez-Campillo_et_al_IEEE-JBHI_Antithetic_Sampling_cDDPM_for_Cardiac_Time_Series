"""VAE denoising configuration (``ml_collections`` ConfigDict).

Faithful migration of ``MAP_VAE/config.py``. Consumed at runtime by the absl
``config_flags.DEFINE_config_file('config', ...)`` mechanism (passed as
``--config=configs/config.py``), not imported. The only non-mechanical change
relative to the original is that the hard-coded local ``working_dir`` default
is now routed through ``cardiac_map_diffusion.paths.experiments_root()`` so no
machine-specific absolute path is baked into the repository; override at runtime
with ``--config.working_dir=...`` exactly as the submission scripts do.
"""

import os
import ml_collections
from cardiac_map_diffusion import paths

def get_config():
    config = ml_collections.ConfigDict()

    config.working_dir = str(paths.experiments_root())
    config.experiment_name = "trial_experiment"
    config.exclude_patients_file = '' # File to exclude patients from training (e.g. for hidden test set)
    config.cluster = False
    config.seed = 0

    config.model = "CNN"
    config.optimizer = "adam" # {adam, sgd}
    config.beta = 0.1
    config.architecture = 5
    config.batch_size = 16
    config.num_epochs = 50
    config.learning_rate = 5e-4
    config.optimizer_scheduler = False
    config.beta_schedule_mode = 'linear' # {linear, sigmoid, cosine, none}
    config.beta_schedule_cycles = 1
    config.beta_schedule_ratio = 0.5
    config.seed = 17
    config.weight_constant = 1
    config.num_folds = 4
    config.split_number = 0
    config.num_workers = 4
    config.prefetch_factor = 2
    config.latent_size = 32
    config.FIGURE_DICT = True
    config.seed_split = 29


    config.noise_type = "gaussian" # ['truncation', 'gaussian', 'spike', 'bwander', 'powerline']
    return config
