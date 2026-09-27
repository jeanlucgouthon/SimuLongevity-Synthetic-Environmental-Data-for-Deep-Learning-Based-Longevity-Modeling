import torch

from src.simudata.distributions import simualate_normal

def gaussian_noise(
    n: int,
    mean: float = 0.0,
    std: float = 1.0
) -> torch.Tensor:
    """
    Generate random noise from a normal distribution.

    The noise is sampled from a normal distribution with the
    specified mean and standard deviation.

    Parameters
    ----------
    n : int
        Number of noise values to generate.
    mean : float, optional
        Mean of the normal distribution.
    std : float, optional
        Standard deviation of the normal distribution. Must be
        strictly positive.

    Returns
    -------
    torch.Tensor
        A one-dimensional tensor containing `n` samples from the
        normal distribution.

    Raises
    ------
    ValueError
        If `std` is not strictly positive.

    Examples
    --------
    >>> epsilon = noise(n=365, mean=0.0, std=1.0)
    >>> epsilon.shape
    torch.Size([365])
    """
    if std <= 0.0:
        raise ValueError("Std must be a positive float.")

    return simualate_normal(n, mean, std)