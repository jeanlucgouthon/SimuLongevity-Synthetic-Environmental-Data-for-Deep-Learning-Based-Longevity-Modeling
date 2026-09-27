
import torch 
from src.simudata.noise import gaussian_noise


def simulate_pm25(
    temperature_data: torch.Tensor,
    humidity_data: torch.Tensor,
    intercept: float = 2.0,
    temperature_slope: float = 0.15,
    humidity_slope: float = 0.1,
    noise_mean: float = 0.0,
    noise_std: float = 0.3,
) -> torch.Tensor:
    """Simulate PM2.5 concentrations from temperature and humidity.

    The simulation is based on a log-linear model in which PM2.5
    concentration increases with humidity, decreases with temperature,
    and is affected by Gaussian random noise:

        log(PM2.5) = intercept
                     + humidity_slope * humidity
                     - temperature_slope * temperature
                     + noise

    Parameters
    ----------
    temperature_data : torch.Tensor
        Temperature values for each time step.
    humidity_data : torch.Tensor
        Relative humidity values for each time step.
    intercept : float, default=2.0
        Intercept of the log-PM2.5 model.
    temperature_slope : float, default=0.02
        Effect of temperature on log-PM2.5. A positive value means that
        PM2.5 decreases as temperature increases.
    humidity_slope : float, default=0.01
        Effect of humidity on log-PM2.5. A positive value means that
        PM2.5 increases as humidity increases.
    noise_mean : float, default=0.0
        Mean of the Gaussian random noise added to log-PM2.5.
    noise_std : float, default=2.0
        Standard deviation of the Gaussian random noise.

    Returns
    -------
    torch.Tensor
        Simulated PM2.5 concentrations, with the same shape, device,
        and dtype as the input data.

    Raises
    ------
    ValueError
        If `temperature_data` and `humidity_data` do not have the same
        shape.
    """
    if temperature_data.shape != humidity_data.shape:
        raise ValueError(
            "temperature_data and humidity_data must have the same shape."
        )

    temperature_scale = (temperature_data - temperature_data.mean()) / temperature_data.std()
    humidity_scale = (humidity_data - humidity_data.mean()) / humidity_data.std()
    noise = gaussian_noise(
        mean=noise_mean,
        std=noise_std,
        n=temperature_data.numel(),
        device=temperature_data.device,
        dtype=temperature_data.dtype,
    ).reshape(temperature_data.shape)

    log_pm25 = (
        intercept
        + humidity_slope * humidity_scale
        - temperature_slope * temperature_scale
        + noise
    )

    return torch.exp(log_pm25)
