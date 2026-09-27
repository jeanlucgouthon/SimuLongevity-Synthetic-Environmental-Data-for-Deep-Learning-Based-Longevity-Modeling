import torch

from src.simudata.distributions import simualate_normal

def gaussian_noise(
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | str = "mps",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate random noise from a Gaussian distribution.

    Parameters
    ----------
    n : int
        Number of noise values to generate.
    mean : float, optional
        Mean of the Gaussian distribution.
    std : float, optional
        Standard deviation of the Gaussian distribution. Must be
        strictly positive.
    device : torch.device or str, optional
        Device on which to create the tensor.
    dtype : torch.dtype, optional
        Data type of the generated tensor.

    Returns
    -------
    torch.Tensor
        A one-dimensional tensor containing `n` samples from the
        Gaussian distribution.

    Raises
    ------
    ValueError
        If `n` is negative or `std` is not strictly positive.

    Examples
    --------
    >>> epsilon = gaussian_noise(n=365, mean=0.0, std=1.0)
    >>> epsilon.shape
    torch.Size([365])
    """

    if n < 0:
        raise ValueError("n must be non-negative.")

    if std <= 0.0:
        raise ValueError("std must be strictly positive.")

    return torch.normal(
        mean=mean,
        std=std,
        size=(n,),
        device=device,
        dtype=dtype,
    )