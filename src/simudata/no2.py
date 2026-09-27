
import torch 
from src.simudata.noise import gaussian_noise


import torch
from src.simudata.noise import gaussian_noise


def simulate_no2(
    temperature_data: torch.Tensor | None,
    humidity_data: torch.Tensor | None,
    intercept: float = 1.5,
    temperature_slope: float = 0.08,
    humidity_slope: float = 0.18,
    noise_mean: float = 0.0,
    noise_std: float = 0.2,
) -> torch.Tensor:
    """
    Simulate nitrogen dioxide (NO₂) concentrations from meteorological data.

    The NO₂ concentration is generated on a log scale using standardized
    temperature and humidity effects, followed by a log-normal transformation.
    The coefficients are chosen to provide behavior distinct from the PM2.5
    simulation while preserving plausible environmental variability.

    Parameters
    ----------
    temperature_data : torch.Tensor | None
        Temperature time series used as an explanatory variable.
    humidity_data : torch.Tensor | None
        Relative humidity time series used as an explanatory variable.
    intercept : float, default=1.5
        Baseline value on the log-concentration scale.
    temperature_slope : float, default=0.08
        Magnitude of the temperature effect. Higher temperatures reduce the
        simulated NO₂ concentration.
    humidity_slope : float, default=0.18
        Magnitude of the humidity effect. Higher humidity increases the
        simulated NO₂ concentration.
    noise_mean : float, default=0.0
        Mean of the Gaussian random noise added on the log scale.
    noise_std : float, default=0.2
        Standard deviation of the Gaussian random noise added on the log scale.

    Returns
    -------
    torch.Tensor
        Simulated positive NO₂ concentration values with the same shape,
        device, and dtype as the input temperature data.

    Raises
    ------
    ValueError
        If temperature_data or humidity_data is None, or if the two tensors
        do not have the same shape.
    """
    if temperature_data is None or humidity_data is None:
        raise ValueError(
            "temperature_data and humidity_data must not be None."
        )

    if temperature_data.shape != humidity_data.shape:
        raise ValueError(
            "temperature_data and humidity_data must have the same shape."
        )

    temperature_scale = (
        temperature_data - temperature_data.mean()
    ) / temperature_data.std()

    humidity_scale = (
        humidity_data - humidity_data.mean()
    ) / humidity_data.std()

    noise = gaussian_noise(
        mean=noise_mean,
        std=noise_std,
        n=temperature_data.numel(),
        device=temperature_data.device,
        dtype=temperature_data.dtype,
    ).reshape(temperature_data.shape)

    log_no2 = (
        intercept
        - temperature_slope * temperature_scale
        + humidity_slope * humidity_scale
        + noise
    )

    return torch.exp(log_no2)