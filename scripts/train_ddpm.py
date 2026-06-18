#!/usr/bin/env python
"""CLI entry point for conditional DDPM training (VAE-aligned).

Thin wrapper around :func:`cardiac_map_diffusion.diffusion.train_ddpm.main`.
Requires the package to be installed (``pip install -e .``). All command-line
arguments are forwarded to the trainer's argparse parser.

Examples
--------
    # paper configuration (see configs/ddpm.yaml and slurm/train_ddpm.sbatch)
    python scripts/train_ddpm.py --experiment final_big_80 --device cuda \
        --seed 17 --seed_split 29 --signal_length 370 --n_prints 5 --lr 0.001 \
        --noise_batchwise True --noise_steps 5000 --noise_schedule quadratic \
        --beta_start 0.0001 --beta_end 0.05 --use_pretrained False \
        --model_small False --feats 80 --num_epochs 400 --batch_size 96 \
        --batch_size_test 192 --normalise True --test_size 0.2 --n_splits 4 \
        --step_size 100 --gamma 0.5

    python scripts/train_ddpm.py --help
"""

from cardiac_map_diffusion.diffusion.train_ddpm import main

if __name__ == "__main__":
    main()
