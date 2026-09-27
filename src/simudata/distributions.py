import torch 


def simualate_normal(n: int,
                     mean:float=0.0, 
                     std:float=1.0
                     ) -> torch.Tensor:
    
    """
    Simulate n observations from a normal distribution.
    X ~ N(mean, std^2)
    """
    return torch.normal(mean=mean,
                        std=std, 
                        size=(n,))

def simualate_lognormal(n: int,
                     mean:float=0.0, 
                     std:float=1.0
                     ) -> torch.Tensor:
    
    """
    Simulate n observations from a lognormal distribution.
    X ~ LogN(mean, std^2)
    """
    dist = torch.distributions.LogNormal(loc=mean,
                        scale=std)
    return dist.sample((n,)) # type: ignore

def simulate_gamma(n: int,
                   shape: float=1.0,
                   rate:float=1.0):
    dist = torch.distributions.Gamma(shape, rate)
    return dist.sample((n,))