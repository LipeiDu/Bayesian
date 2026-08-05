"""Utilities shared by posterior consumers."""

import numpy as np
import numpy.typing as npt


def flatten_chain_samples(
    chain: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return 2D posterior samples from emcee or flat weighted samplers."""
    if chain.ndim == 3:
        return chain.reshape((chain.shape[0] * chain.shape[1], chain.shape[2]))
    if chain.ndim == 2:
        return chain
    raise ValueError(f"Unsupported chain shape {chain.shape}; expected 2D or 3D samples.")
