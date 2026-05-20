from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def load_realnvp_2d_from_payload(payload: dict[str, Any]) -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "Sequential flow prior support requires PyTorch in the Bayesian environment."
        ) from exc

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    class _CouplingMLP(nn.Module):
        def __init__(self, *, hidden_features: int, hidden_layers: int) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_features = 2
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(in_features, hidden_features), nn.ReLU()])
                in_features = hidden_features
            layers.append(nn.Linear(in_features, 4))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> tuple[Any, Any]:
            shift, log_scale = self.net(x).chunk(2, dim=-1)
            return shift, log_scale

    class _AffineCoupling2D(nn.Module):
        def __init__(self, mask: npt.NDArray[np.float64], *, hidden_features: int, hidden_layers: int, scale_limit: float) -> None:
            super().__init__()
            self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
            self.nn = _CouplingMLP(hidden_features=hidden_features, hidden_layers=hidden_layers)
            self.scale_limit = float(scale_limit)

        def forward(self, x: Any) -> tuple[Any, Any]:
            x_masked = x * self.mask
            shift, log_scale = self.nn(x_masked)
            inverse_mask = 1.0 - self.mask
            shift = shift * inverse_mask
            log_scale = torch.tanh(log_scale) * self.scale_limit * inverse_mask
            y = x_masked + inverse_mask * (x * torch.exp(log_scale) + shift)
            log_det = torch.sum(log_scale, dim=-1)
            return y, log_det

        def inverse(self, y: Any) -> tuple[Any, Any]:
            y_masked = y * self.mask
            shift, log_scale = self.nn(y_masked)
            inverse_mask = 1.0 - self.mask
            shift = shift * inverse_mask
            log_scale = torch.tanh(log_scale) * self.scale_limit * inverse_mask
            x = y_masked + inverse_mask * (y - shift) * torch.exp(-log_scale)
            log_det = -torch.sum(log_scale, dim=-1)
            return x, log_det

    class _RealNVP2D(nn.Module):
        def __init__(self, *, n_coupling_layers: int, hidden_features: int, hidden_layers: int, scale_limit: float) -> None:
            super().__init__()
            masks = [np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float)]
            self.layers = nn.ModuleList(
                [
                    _AffineCoupling2D(
                        masks[i % 2],
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        scale_limit=scale_limit,
                    )
                    for i in range(n_coupling_layers)
                ]
            )

        def log_prob(self, x: Any) -> Any:
            z = x
            log_det_total = torch.zeros(x.shape[0], device=x.device)
            for layer in reversed(self.layers):
                z, log_det = layer.inverse(z)
                log_det_total += log_det
            base_log_prob = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=-1)
            return base_log_prob + log_det_total

        def sample(self, n_samples: int) -> Any:
            z = torch.randn((n_samples, 2), dtype=torch.float32)
            x = z
            for layer in self.layers:
                x, _ = layer.forward(x)
            return x

    architecture = payload["architecture"]
    model = _RealNVP2D(
        n_coupling_layers=int(architecture["n_coupling_layers"]),
        hidden_features=int(architecture["hidden_features"]),
        hidden_layers=int(architecture["hidden_layers"]),
        scale_limit=float(architecture["scale_limit"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
