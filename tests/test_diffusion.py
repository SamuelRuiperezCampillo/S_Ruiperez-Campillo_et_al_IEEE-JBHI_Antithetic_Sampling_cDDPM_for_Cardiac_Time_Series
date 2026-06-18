"""Light, data-free checks of the diffusion core (no training, no cohort).

Constructing the ``Diffusion`` process exercises the noise-schedule construction
(``prepare_noise_schedule``) and tensor allocation. Note: ``Diffusion`` forces its
own device (``'cuda' if torch.cuda.is_available() else 'cpu'``), so this runs on
whatever hardware the test host has.
"""

import pytest

torch = pytest.importorskip("torch")

from cardiac_map_diffusion.diffusion.ddpm_conditional import Diffusion


def _build(noise_steps=10, schedule="quadratic"):
    # Positional call matches the trainer's usage:
    # Diffusion(noise_steps, beta_start, beta_end, signal_len, noise_schedule, device)
    return Diffusion(noise_steps, 1e-4, 0.05, 370, schedule, "cpu")


@pytest.mark.parametrize("schedule", ["linear", "quadratic"])
def test_diffusion_constructs(schedule):
    diffusion = _build(noise_steps=10, schedule=schedule)
    assert diffusion is not None
    # The beta schedule should have one entry per diffusion step.
    assert int(diffusion.beta.shape[0]) == 10


def test_networks_instantiate():
    from cardiac_map_diffusion.diffusion.denoising_net import ConditionalModel
    from cardiac_map_diffusion.diffusion.denoising_net_small import ConditionalModelSmall

    assert ConditionalModel(8) is not None
    assert ConditionalModelSmall(8) is not None
