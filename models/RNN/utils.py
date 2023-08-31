import copy
from typing import Tuple, Union

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


def add_prefix(
    batch: Tuple[Tensor, Union[Tensor, None]],
    *,
    return_batch: bool = True,
) -> Tuple[Union[Tuple[Tensor, Tensor], Tensor], Tensor]:
    """
    _summary_

    Parameters
    ----------
    batch : Tuple[Tensor, Tensor | None]
        _description_
    return_batch : bool, optional
        _description_, by default True

    Returns
    -------
    Tuple[Tuple[Tensor, Tensor] | Tensor, Tensor]:
        _description_
    """

    strokes, text = batch
    strokes_copy = copy.deepcopy(strokes)
    batch_size, _, input_len = strokes.size()

    prefix = torch.zeros([batch_size, 1, input_len], device=strokes.device)
    strokes = torch.cat([prefix, strokes[:, :-1]], dim=1)

    return ((strokes, text), strokes_copy) if return_batch else (strokes, strokes_copy)


def get_initial_states(
    batch_size: int, hidden_size: int, *, device: torch.device
) -> Tuple[Tuple[Tensor, Tensor], ...]:
    """
    _summary_

    Parameters
    ----------
    batch_size : int
        _description_
    hidden_size : int
        _description_
    device : torch.device
        _description_

    Returns
    -------
    Tuple[Tuple[Tensor, Tensor], ...]
        _description_
    """

    h_0 = torch.zeros(1, batch_size, hidden_size, device=device)
    c_0 = torch.zeros_like(h_0, device=device)

    return (h_0, c_0), (h_0, c_0), (h_0, c_0)


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

    pi, mu, sigma, rho, eos = mixtures
    num_components = pi.size(-1)

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

    eos_flag = 1 if eos > 0.5 else 0

    return torch.tensor([x, y, eos_flag], device=pi.device)


def bivariate_gaussian(
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
