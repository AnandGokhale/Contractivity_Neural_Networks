import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _sym_inv_sqrt(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """M^{-1/2} for a symmetric PD matrix via eigen-decomposition."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_sqrt = 1.0 / eigvals.clamp(min=eps).sqrt()
    return (eigvecs * inv_sqrt.unsqueeze(-2)) @ eigvecs.mT


class FRNN(nn.Module):
    def __init__(self, nx_orig, nu, nx_ext=60, nonlin=nn.Tanh):
        super().__init__()
        self.nx_orig = nx_orig
        self.nx_ext  = nx_ext
        self.nu      = nu

        self.in_features  = nx_ext + nu
        self.out_features = nx_ext
        n = nx_ext

        self.c = 0.5
        self.epsilon = 0.01


        self.q_raw  = nn.Parameter(torch.zeros(n))
        self.X_free = nn.Parameter(torch.randn(n, n) * 0.01)
        self.V_free = nn.Parameter(torch.eye(n) + 0.01 * torch.randn(n, n))


        # self.A   = nn.Linear(nx_ext, nx_ext, bias=False)
        self.B   = nn.Linear(nu,     nx_ext, bias=False)
        self.b   = nn.Parameter(torch.zeros(nx_ext))
        self.C   = nn.Linear(nx_ext, nx_orig, bias=False) # converts nx_ext back to nx_orig for comparison to data
        self.encoder = nn.Linear(nx_orig, nx_ext)   # learned x0 -> z0
        
        self.phi = nonlin()

    def _get_W(self):
        n      = self.nx_ext
        device = self.q_raw.device

        q   = F.softplus(self.q_raw)                           # (n,)  strictly positive
        M   = torch.eye(n, device=device) + self.X_free.mT @ self.X_free
        S   = self.X_free @ _sym_inv_sqrt(M)                   # S^T S <= I
        VtV = self.V_free.mT @ self.V_free + self.epsilon * torch.eye(n, device=device)

        term1 = 2 * np.sqrt(1 - self.c) * (q.unsqueeze(-1) * (S @ VtV))
        term2 = (q**2).unsqueeze(-1) * (VtV @ VtV)
        return term1 - term2


    def forward(self, z, u, t=None):
        # zu: (..., nx_ext + nu)
        #z = zu[..., :self.nx_ext]
        #u = zu[..., self.nx_ext:]
        # 
        W = self._get_W()

        # return -z + self.phi(self.A(z) + self.B(u) + self.b)

        return -z + self.phi(z @ W.mT + self.B(u) + self.b)    

class StateLifter(nn.Module):
    """
    Lifts x (nx_orig,) -> z (nx_ext,) by padding with zeros.
    Or use a learned encoder by setting learned=True.
    """
    def __init__(self, nx_orig, nx_ext, learned=False):
        super().__init__()
        self.nx_orig = nx_orig
        self.nx_ext  = nx_ext
        if learned:
            self.encoder = nn.Linear(nx_orig, nx_ext)
        else:
            self.encoder = None

    def forward(self, x):
        if self.encoder is not None:
            return self.encoder(x)
        # Zero-pad: z = [x, 0, 0, ..., 0]
        pad = torch.zeros(*x.shape[:-1], self.nx_ext - self.nx_orig, device=x.device)
        return torch.cat([x, pad], dim=-1)