"""Autoregressive rollout wrapper for a single-step PDE dynamics operator.

Adapts a frozen single-step operator (e.g. ``FNOObserver``) into the standard
DDIS surrogate interface ``surrogate(noisy_state) -> predicted_observation`` —
except the predicted observation is now a T-step time series of wall-boundary
fields, produced by autoregressive rollout with known (noiseless) actions and
forcings.

Gradient flow: the wrapped operator's weights are frozen, but autograd still
propagates through every rollout step back to ``x0``.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt


class DynamicsSurrogate(nn.Module):
    """Wraps a single-step dynamics model into a T-step rollout surrogate.

    Call signature::

        surrogate(x0, actions, forcings) -> wall_obs

        x0:       (B, 4, Nx, Ny, Nz)        noisy initial state, ``requires_grad``
        actions:  (B, T, Nx, 1, Nz)         noiseless wall actuations
        forcings: (B, T)                    noiseless scalar forcings
        wall_obs: (B, T, 3, Nx, 1, Nz)      [du/dy, dw/dy, p] at y=0
    """

    def __init__(self, dynamics_model: nn.Module, dy: float, use_checkpoint: bool = False):
        """
        Args:
            dynamics_model: callable matching FNOObserver's signature:
                ``f(vel_pres, forcing, action) -> (B, 4, Nx, Ny, Nz)`` where
                inputs are ``(B, 1, 4, Nx, Ny, Nz), (B, 1), (B, 1, Nx, 1, Nz)``.
                Assumed trained with ``n_history=1``.
            dy: spacing between wall (y=0) and first interior point (y=1);
                used for the one-sided shear-stress finite difference.
            use_checkpoint: trade compute for activation memory across the
                rollout (useful for long T).
        """
        super().__init__()
        self.dynamics_model = dynamics_model
        self.dynamics_model.requires_grad_(False)
        self.dynamics_model.eval()
        self.dy = float(dy)
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def _wall_obs(state: torch.Tensor, inv_dy: float) -> torch.Tensor:
        """Extract (du/dy, dw/dy, p) at y=0 from a (B,4,Nx,Ny,Nz) state.

        All intermediate tensors are views via slicing; only the final
        ``torch.cat`` allocates.
        """
        u_y0 = state[:, 0:1, :, 0:1, :]
        u_y1 = state[:, 0:1, :, 1:2, :]
        w_y0 = state[:, 2:3, :, 0:1, :]
        w_y1 = state[:, 2:3, :, 1:2, :]
        p_y0 = state[:, 3:4, :, 0:1, :]
        du_dy = (u_y1 - u_y0) * inv_dy
        dw_dy = (w_y1 - w_y0) * inv_dy
        return torch.cat([du_dy, dw_dy, p_y0], dim=1)  # (B, 3, Nx, 1, Nz)

    def _step(self, state: torch.Tensor, forcing_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        # ``unsqueeze(1)`` is a stride-only view; no copy.
        return self.dynamics_model(state.unsqueeze(1), forcing_t, action_t)

    def forward(self, x0: torch.Tensor, actions: torch.Tensor, forcings: torch.Tensor) -> torch.Tensor:
        T = actions.shape[1]
        inv_dy = 1.0 / self.dy

        # Match the operator's parameter dtype to avoid implicit upcasts inside.
        target_dtype = next(self.dynamics_model.parameters()).dtype
        if x0.dtype != target_dtype:
            x0 = x0.to(target_dtype)
        if actions.dtype != target_dtype:
            actions = actions.to(target_dtype)
        if forcings.dtype != target_dtype:
            forcings = forcings.to(target_dtype)

        state = x0
        preds = []
        for t in range(T):
            # Slicing returns views; no per-step copies of the control tensors.
            a_t = actions[:, t : t + 1]
            f_t = forcings[:, t : t + 1]
            if self.use_checkpoint:
                state = _ckpt(self._step, state, f_t, a_t, use_reentrant=False)
            else:
                state = self._step(state, f_t, a_t)
            preds.append(self._wall_obs(state, inv_dy))

        # Single allocation at the end. Gradients flow back through every step.
        return torch.stack(preds, dim=1)  # (B, T, 3, Nx, 1, Nz)
