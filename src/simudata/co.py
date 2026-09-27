import torch
from src.simudata.noise import gaussian_noise


def simulate_co(
    temperature_data: torch.Tensor | None,
    humidity_data: torch.Tensor | None,
    intercept: float = 1.4,
    temperature_slope: float = -0.12,
    humidity_slope: float = 0.08,
    humidity_log_slope: float = 0.10,
    temperature_humidity_slope: float = 0.04,
    noise_mean: float = 0.0,
    noise_std: float = 0.20,
) -> torch.Tensor:
    """
    Simulate carbon monoxide (CO) concentrations from meteorological data.

    The CO concentration is generated on a log scale using standardized
    temperature and humidity effects. A logarithmic transformation of
    humidity is included to introduce a nonlinear and saturating effect,
    while an interaction between temperature and humidity allows the
    temperature effect to vary with atmospheric moisture.

    The simulated log-concentration is defined as:

        log(CO) = β₀ + β_T T + β_H H
                  + β_log log(1 + H*) + β_TH (T × H) + ε,

    where T and H are standardized temperature and humidity, respectively,
    H* is the rescaled humidity, and ε is Gaussian noise.

    Parameters
    ----------
    temperature_data : torch.Tensor | None
        Temperature time series used as an explanatory variable.
    humidity_data : torch.Tensor | None
        Relative humidity time series used as an explanatory variable.
    intercept : float, default=1.4
        Baseline value on the log-concentration scale.
    temperature_slope : float, default=-0.12
        Linear temperature effect. Higher temperatures reduce the simulated
        CO concentration.
    humidity_slope : float, default=0.08
        Linear humidity effect.
    humidity_log_slope : float, default=0.10
        Nonlinear humidity effect based on a logarithmic transformation.
    temperature_humidity_slope : float, default=0.04
        Interaction effect between standardized temperature and humidity.
    noise_mean : float, default=0.0
        Mean of the Gaussian random noise added on the log scale.
    noise_std : float, default=0.20
        Standard deviation of the Gaussian random noise added on the log scale.

    Returns
    -------
    torch.Tensor
        Simulated positive CO concentration values with the same shape,
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

    humidity_positive = humidity_scale - humidity_scale.min() + 1.0

    humidity_log = torch.log1p(humidity_positive)

    temperature_humidity = temperature_scale * humidity_scale

    noise = gaussian_noise(
        mean=noise_mean,
        std=noise_std,
        n=temperature_data.numel(),
        device=temperature_data.device,
        dtype=temperature_data.dtype,
    ).reshape(temperature_data.shape)

    log_co = (
        intercept
        + temperature_slope * temperature_scale
        + humidity_slope * humidity_scale
        + humidity_log_slope * humidity_log
        + temperature_humidity_slope * temperature_humidity
        + noise
    )

    return torch.exp(log_co)