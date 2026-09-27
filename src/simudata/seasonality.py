import torch

def sinusoidal_seasonality(
    t: torch.Tensor,
    period: float,
    amplitude: float=1.0,
    phase: float=0.0
) -> torch.Tensor:
    """ 
    Generate a sinusoidal seasonal component.
    
    Parameters
    ----------
    t : torch.Tensor
        Time points.
    period : float
        Seasonal period.
    amplitude : float
        Amplitude of the seasonal variation.
    phase : float
        Phase shift.

    Returns
    -------
    torch.Tensor
        Seasonal component.
    """
    return amplitude * torch.sin(
        2 * torch.pi * t / period + phase
                                 )

def harmonic_seasonality(
    t: torch.Tensor,
    period: float,
    amplitudes: torch.Tensor,
    phases: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    Generate a seasonal component using a sum of harmonic functions.

    The seasonal component is defined as

        $S(t) = sum_{k=1}^{K} A_k \times
               sin(\frac{2 \times pi \times k \times t}{period} + phi_k)$,

    where A_k and phi_k are the amplitude and phase of the k-th
    harmonic, respectively.

    Parameters
    ----------
    t : torch.Tensor
        Time points at which the seasonal component is evaluated.
    period : float
        Fundamental period of the seasonal cycle.
        For example, use 365 for yearly seasonality in daily data.
    amplitudes : torch.Tensor
        Amplitudes of the harmonic components. The number of
        harmonics is determined by the length of this tensor.
    phases : torch.Tensor or None, optional
        Phase shifts of the harmonic components. If None, all
        phase shifts are set to zero.

    Returns
    -------
    torch.Tensor
        Seasonal component evaluated at each time point in `t`.

    Examples
    --------
    >>> t = torch.arange(365)
    >>> amplitudes = torch.tensor([0.3, 0.1])
    >>> seasonality = harmonic_seasonality(t, 365, amplitudes)
    """
    k = torch.arange(
        1,
        len(amplitudes) + 1,
        device=t.device,
    )

    if phases is None:
        phases = torch.zeros_like(amplitudes)

    return torch.sum(
        amplitudes[:, None]
        * torch.sin(
            2 * torch.pi * k[:, None] * t[None, :] / period
            + phases[:, None]
        ),
        dim=0,
    )