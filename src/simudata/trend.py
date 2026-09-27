import torch


def linear_trend(
    t: torch.Tensor,
    intercept: float = 0.0,
    slope: float = 1.0,
) -> torch.Tensor:
    """
    Generate a linear trend over time.

    The trend is defined as

        T(t) = intercept + slope * t.

    Parameters
    ----------
    t : torch.Tensor
        Time points at which the trend is evaluated.
    intercept : float, optional
        Value of the trend at t = 0.
    slope : float, optional
        Rate of change per unit of time.

    Returns
    -------
    torch.Tensor
        Linear trend evaluated at each time point in `t`.
    """
    return intercept + slope * t


def exponential_trend(
    t: torch.Tensor,
    initial_value: float = 1.0,
    growth_rate: float = 0.01,
) -> torch.Tensor:
    """
    Generate an exponential trend over time.

    The trend is defined as

        T(t) = initial_value * exp(growth_rate * t).

    Parameters
    ----------
    t : torch.Tensor
        Time points at which the trend is evaluated.
    initial_value : float, optional
        Value of the trend at t = 0.
    growth_rate : float, optional
        Exponential growth rate per unit of time.
        A positive value produces growth, while a negative value
        produces exponential decay.

    Returns
    -------
    torch.Tensor
        Exponential trend evaluated at each time point in `t`.
    """
    return initial_value * torch.exp(growth_rate * t)