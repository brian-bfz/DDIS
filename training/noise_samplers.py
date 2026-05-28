import torch


class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError()


class RBFKernel(NoiseSampler):
    @torch.no_grad()
    def __init__(self, in_channels, Ln1, Ln2, Ln3=None, scale=1, eps=0.01, device=None):
        self.in_channels = in_channels
        self.dims = [Ln1, Ln2] if Ln3 is None else [Ln1, Ln2, Ln3]
        self.device = device

        self.factors = []
        for n in self.dims:
            xs = torch.linspace(0, 1, steps=n + 1, device=device)[:-1].unsqueeze(-1)  # (n, 1)
            C = torch.exp(-torch.cdist(xs, xs)**2 / (2 * scale**2))
            C.diagonal().add_(eps**2)
            self.factors.append(torch.linalg.cholesky(C))

    @torch.no_grad()
    def sample(self, N):
        Z = torch.randn(N, self.in_channels, *self.dims, device=self.device)

        ndim = Z.dim()
        for d, L in enumerate(self.factors, start=2):
            perm = list(range(ndim))
            perm[d], perm[-1] = perm[-1], perm[d]
            Z = Z.permute(perm) @ L.T
            Z = Z.permute(perm)

        return Z
