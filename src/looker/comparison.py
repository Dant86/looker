"""Calculate comparison metrics."""

import torch


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Calculate the Jensen-Shannon Divergence of two distributions.
    
    Args:
        p: A tensor of shape (n,).
        q: A tensor of shape (n,).
        
    Returns:
        A tensor of shape (1,).
    """
    mixture = (p + q) / 2

    kl_p_mixture = p @ (torch.log(p) - torch.log(mixture))
    kl_q_mixture = q @ (torch.log(q) - torch.log(mixture))

    return (kl_p_mixture + kl_q_mixture) / 2
