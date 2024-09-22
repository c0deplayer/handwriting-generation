import torch


def get_device() -> "str":
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    return "cpu"
