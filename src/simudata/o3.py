import torch
from src.simudata.noise import gaussian_noise


import torch
from src.simudata.noise import gaussian_noise


def simulate_o3(
    temperature_data: torch.Tensor | None,
    humidity_data: torch.Tensor | None,
    intercept: float = 1.8,
    temperature_slope: float = 0.20,
    humidity_slope: float = -0.10,
    temperature_squared_slope: float = 0.06,
    temperature_humidity_slope: float = -0.05,
    noise_mean: float = 0.0,
    noise_std: float = 0.18,
) -> torch.Tensor:
    """
    Simulate ozone (O₃) concentrations from meteorological data.

    The O₃ concentration is generated on a log scale using standardized
    temperature and humidity effects. In addition to linear effects, the
    simulation includes a quadratic temperature term and a temperature-
    humidity interaction to represent nonlinear meteorological effects on
    ozone concentrations.

    The simulated log-concentration is defined as:

        log(O₃) = β₀ + β_T T + β_H H + β_T² T²
                  + β_TH (T × H) + ε,

    where T and H are standardized temperature and humidity, respectively,
    and ε is Gaussian noise.

    Parameters
    ----------
    temperature_data : torch.Tensor | None
        Temperature time series used as an explanatory variable.
    humidity_data : torch.Tensor | None
        Relative humidity time series used as an explanatory variable.
    intercept : float, default=1.8
        Baseline value on the log-concentration scale.
    temperature_slope : float, default=0.20
        Linear temperature effect. Higher temperatures increase the
        simulated O₃ concentration.
    humidity_slope : float, default=-0.10
        Linear humidity effect. Higher humidity tends to reduce the
        simulated O₃ concentration.
    temperature_squared_slope : float, default=0.06
        Nonlinear temperature effect represented by the squared standardized
        temperature.
    temperature_humidity_slope : float, default=-0.05
        Interaction effect between standardized temperature and humidity.
    noise_mean : float, default=0.0
        Mean of the Gaussian random noise added on the log scale.
    noise_std : float, default=0.18
        Standard deviation of the Gaussian random noise added on the log scale.

    Returns
    -------
    torch.Tensor
        Simulated positive O₃ concentration values with the same shape,
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

    temperature_squared = temperature_scale**2

    temperature_humidity = temperature_scale * humidity_scale

    noise = gaussian_noise(
        mean=noise_mean,
        std=noise_std,
        n=temperature_data.numel(),
        device=temperature_data.device,
        dtype=temperature_data.dtype,
    ).reshape(temperature_data.shape)

    log_o3 = (
        intercept
        + temperature_slope * temperature_scale
        + humidity_slope * humidity_scale
        + temperature_squared_slope * temperature_squared
        + temperature_humidity_slope * temperature_humidity
        + noise
    )

    return torch.exp(log_o3)