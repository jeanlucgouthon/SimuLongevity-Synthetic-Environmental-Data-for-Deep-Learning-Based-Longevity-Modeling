import torch 
from src.simudata.noise import gaussian_noise

def simulate_humidity(
    temperature_data: torch.Tensor,
    intercept: float = 75.0,
    slope: float = 0.5,
    temperature_bias: float = 15.0,
    noise_mean: float = 0.0,
    noise_std: float = 5.0,
    min_humidity: float = 0.0,
    max_humidity: float = 100.0,
) -> torch.Tensor:
    r"""
    Simulate relative humidity (%) from temperature data.

    The humidity is modeled as an approximately linear decreasing
    function of temperature, with Gaussian noise:

        $H_t = intercept - slope * (T_t - temperature_bias) + \varepsilon_t$

    where $\varepsilon_t ~ N(noise\_mean, noise\_std^2)$.

    Parameters
    ----------
    temperature_data : torch.Tensor
        Temperature values used to generate humidity.
    intercept : float, default=75.0
        Expected humidity (%) when temperature equals `temperature_bias`.
    slope : float, default=0.5
        Decrease in humidity (%) for a one-degree increase in temperature.
    temperature_bias : float, default=15.0
        Reference temperature (°C).
    noise_mean : float, default=0.0
        Mean of the Gaussian noise.
    noise_std : float, default=5.0
        Standard deviation of the Gaussian noise.
    min_humidity : float, default=0.0
        Minimum physically valid relative humidity (%).
    max_humidity : float, default=100.0
        Maximum physically valid relative humidity (%).

    Returns
    -------
    torch.Tensor
        Simulated relative humidity values in [0, 100] (%).
    """

    noise = gaussian_noise(
        mean=noise_mean,
        std=noise_std,
        n=temperature_data.shape[0],
        device=temperature_data.device,
        dtype=temperature_data.dtype,
    ) 

    humidity = (
        intercept
        - slope * (temperature_data - temperature_bias)
        + noise
    )

    return humidity.clamp(min_humidity, max_humidity)
    