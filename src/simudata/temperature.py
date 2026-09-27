import torch

from src.simudata.distributions import simualate_lognormal, simualate_normal
from src.simudata.seasonality import sinusoidal_seasonality
from src.simudata.trend import linear_trend, exponential_trend
from src.simudata.noise import gaussian_noise

def simulate_temperature(
    n: int = 10000,
    intercept: float = 15.0,
    amplitude: float = 10.0,
    period: int = 365,
    phase: float = 0.0,
    include_trend: bool = True,
    type_of_trend: str = "linear",
    trend_slope: float = 0.0,
    trend_initial_value: float = 0.0,
    trend_growth_rate: float = 0.0,
    noise_mean: float = 0.0,
    noise_std: float = 3.0,
    device:torch.device | str ="mps",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Simulate temperature observations with seasonality, trend,
    and random noise.

    The simulated temperature is defined as

        T(t) = intercept + S(t) + G(t) + epsilon(t),

    where S(t) is a seasonal component, G(t) is an optional trend,
    and epsilon(t) is Gaussian noise.

    Parameters
    ----------
    n : int, optional
        Number of observations to simulate.
    intercept : float, optional
        Baseline temperature.
    amplitude : float, optional
        Amplitude of the seasonal component.
    period : int, optional
        Period of the seasonal cycle.
    phase : float, optional
        Phase shift of the seasonal component, in radians.
    include_trend : bool, optional
        Whether to include a trend in the simulated temperature.
    type_of_trend : str, optional
        Type of trend. Supported values are ``"linear"`` and
        ``"exponential"``.
    trend_slope : float, optional
        Slope of the linear trend.
    trend_initial_value : float, optional
        Initial value of the exponential trend.
    trend_growth_rate : float, optional
        Growth rate of the exponential trend.
    noise_mean : float, optional
        Mean of the Gaussian noise.
    noise_std : float, optional
        Standard deviation of the Gaussian noise.

    Returns
    -------
    torch.Tensor
        Simulated temperature observations.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if period <= 0:
        raise ValueError("period must be a positive integer.")

    if type_of_trend not in {"linear", "exponential"}:
        raise ValueError(
            "type_of_trend must be either 'linear' or 'exponential'."
        )

    t = torch.arange(n, dtype=torch.float32).to(device=device, dtype=dtype)

    # Seasonal component
    seasonality = sinusoidal_seasonality(
        t=t,
        period=period,
        amplitude=amplitude,
        phase=phase,
    ).to(device=device, dtype=dtype)


    # Trend component
    trend = torch.zeros_like(t).to(device=device, dtype=dtype)

    if include_trend:
        if type_of_trend == "linear":
            trend = linear_trend(
                t=t,
                intercept=0.0,
                slope=trend_slope,
            ).to(device=device, dtype=dtype)

        else:
            trend = exponential_trend(
                t=t,
                initial_value=trend_initial_value,
                growth_rate=trend_growth_rate,
            ).to(device=device, dtype=dtype)

    # Random noise
    epsilon = gaussian_noise(
        n=n,
        mean=noise_mean,
        std=noise_std,
        dtype=dtype,
        device=device
    )

    return intercept + seasonality + trend + epsilon