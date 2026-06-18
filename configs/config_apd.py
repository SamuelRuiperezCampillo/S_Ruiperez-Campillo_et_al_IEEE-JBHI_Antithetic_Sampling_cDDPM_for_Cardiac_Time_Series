import ml_collections

from cardiac_map_diffusion import paths

def get_config():
    config_apd = ml_collections.ConfigDict()
    config_apd.working_dir = str(paths.experiments_root())
    config_apd.experiment_name = "svm_apd30"
    config_apd.random_seed = 17
    config_apd.cluster = True
    config_apd.label_apd = 'APD30_gs'
    config_apd.perc_minimum = 0.5
    config_apd.perc_maximum = 99.5
    config_apd.num_folds = 4

    return config_apd
