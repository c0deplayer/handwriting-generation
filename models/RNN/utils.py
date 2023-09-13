from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def reshape_down(batch: Tensor, ground_true: Tensor = None) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    batch : Tensor
        _description_
    ground_true : Tensor, optional
        _description_, by default None

    Returns
    -------
    Tensor
        _description_

    Raises
    ------
    RuntimeError
        _description_
    """

    if ground_true is None:
        return rearrange(batch, "b h w -> (b h) w")

    if batch.size(0) != ground_true.size(0) or batch.size(1) != ground_true.size(1):
        raise RuntimeError(
            f"Expected batch to be of shape {ground_true.size()}, got {batch.size()}"
        )

    return rearrange(
        batch, "b h w -> (b h) w", b=ground_true.size(0), h=ground_true.size(1)
    )


def get_initial_states(
    n_layers: int, batch_size: int, hidden_size: int, *, device: torch.device
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    _summary_

    Parameters
    ----------
    batch_size : int
        _description_
    hidden_size : int
        _description_
    n_layers : int
        _description_
    device : torch.device
        _description_

    Returns
    -------
    Tuple[Tuple[Tensor, Tensor], ...]
        _description_
    """

    h = torch.zeros(n_layers, batch_size, hidden_size, device=device)
    c = torch.zeros_like(h, device=device)

    return (h, c), (h, c), (h, c)


def get_mean_predictions(mixtures: Tuple[Tensor, ...], *, stochastic: bool) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    mixtures : Tuple[Tensor, ...]
        _description_
    stochastic : bool
        _description_

    Returns
    -------
    Tensor
        _description_
    """

    pi, mu, sigma, rho, eos_flag = mixtures
    num_components = len(pi)

    component = torch.multinomial(pi, 1).item() if stochastic else pi.cpu().argmax()

    mu_1 = mu[component]
    mu_2 = mu[component + num_components]

    sigma_1 = sigma[component]
    sigma_2 = sigma[component + num_components]

    rho = rho[component]

    x, y = mu_1, mu_2

    if stochastic:
        covariance_xy = rho * sigma_1 * sigma_2
        covariance_matrix = torch.tensor(
            [[sigma_1**2, covariance_xy], [covariance_xy, sigma_2**2]],
            device=sigma.device,
        )

        loc = torch.tensor([mu_1.item(), mu_2.item()], device=mu.device)
        value = torch.distributions.MultivariateNormal(loc, covariance_matrix).sample()

        x, y = value[0].item(), value[1].item()

    out = torch.tensor([x, y, eos_flag], device=pi.device)

    return rearrange(out, "v -> 1 v")


def bi_variate_gaussian(
    x: Tensor, y: Tensor, mixtures: Tuple[Tensor, ...], *, eps: float = 1e-16
) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    x : Tensor
        _description_
    y : Tensor
        _description_
    mixtures : Tuple[Tensor, ...]
        _description_
    eps : float, optional
        _description_, by default 1e-16

    Returns
    -------
    Tensor
        _description_
    """

    mu_1, mu_2, sigma_1, sigma_2, rho = mixtures

    x_diff = ((x - mu_1) / (sigma_1 + eps)) ** 2.0
    y_diff = ((y - mu_2) / (sigma_2 + eps)) ** 2.0
    xy_diff = 2.0 * rho * (x - mu_1) * (y - mu_2) / (sigma_1 * sigma_2 + eps)
    z = x_diff + y_diff - xy_diff

    rho_squared = 1.0 - rho**2
    exp = torch.exp(-z / (2.0 * rho_squared + eps))
    # noinspection PyTypeChecker
    denominator = 2.0 * torch.pi * sigma_1 * sigma_2 * torch.sqrt(rho_squared) + eps

    # noinspection PyTypeChecker
    return exp / denominator


# * https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20 *
def truncated_normal_(tensor: Tensor, *, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """
    _summary_

    Parameters
    ----------
    tensor : Tensor
        _description_
    mean : float, optional
        _description_, by default 0.0
    std : float, optional
        _description_, by default 1.0

    Returns
    -------
    Tensor
        _description_
    """

    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

    return tensor
