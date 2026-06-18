"""Conditional DDPM core: forward/reverse process, denoising network, and training.

* :mod:`~cardiac_map_diffusion.diffusion.ddpm_conditional` -- the ``Diffusion``
  process, including ``inference`` (crude Monte-Carlo) and ``inference_antithetic``
  (the deployed 2-shot antithetic-variates sampler).
* :mod:`~cardiac_map_diffusion.diffusion.denoising_net` /
  :mod:`~cardiac_map_diffusion.diffusion.denoising_net_small` -- the conditional
  noise-prediction networks.
* :mod:`~cardiac_map_diffusion.diffusion.train_ddpm` -- the VAE-aligned trainer.
"""
