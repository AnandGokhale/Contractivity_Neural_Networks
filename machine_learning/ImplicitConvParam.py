import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Callable, Optional


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


from ImplicitCell import ImplicitCell


def build_mlp(input_dim, output_dim, hidden_dim=16, dtype=torch.float32):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, dtype=dtype),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim, dtype=dtype)
    )

class ImplicitParameterizerConvolutional(nn.Module):
    def __init__(self, input_dim, output_dim, n, hidden_dim=128,
                 activation=torch.tanh, epsilon: float = 0.001,
                 solver_params: dict = {},
                 dtype=torch.float32):
        super().__init__()

        self.n = n
        self.epsilon = epsilon
        self.c = 0.001
        self.dtype = dtype
        self.activation = activation

        # CNN Feature Extractor
        self.trunk = nn.Sequential(
            # Block 1: 3, 32, 32 -> 64, 16, 16
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 64, 16, 16 -> 128, 8, 8 (Depthwise Separable)
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32), # Depthwise
            nn.Conv2d(32, 64, kernel_size=1),                     # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 128, 8, 8 -> 128, 4, 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.3),          # add this
            # 128 * 4 * 4 = 2048. Reducing hidden_dim saves a lot in the MLP nets.
            nn.Linear(1024, hidden_dim), 
            nn.ReLU()
        )



        # Parameter Generators
        self.net_q = build_mlp(hidden_dim, n, dtype=dtype)
        self.net_Z = build_mlp(hidden_dim, n * n, dtype=dtype)
        self.net_Y = build_mlp(hidden_dim, n * n, dtype=dtype)
        self.net_V = build_mlp(hidden_dim, n * n, dtype=dtype)
        self.net_bias = build_mlp(hidden_dim, n, dtype=dtype)

        self.implicit_cell = ImplicitCell(n, activation, solver_params)
        self.output_layer = nn.Linear(n, output_dim, dtype=dtype)

        # Initialize weights to be small to prevent initial explosion
        self._init_weights()

    def _init_weights(self):
        # Initialize the last layers of parameter generators to near-zero
        # This ensures the implicit layer starts close to Identity/Zero behavior
        for m in [self.net_q[-1], self.net_Z[-1], self.net_Y[-1], self.net_V[-1], self.net_bias[-1]]:
            nn.init.uniform_(m.weight, -0.01, 0.01)
            nn.init.zeros_(m.bias)

    def _get_params(self, x):
        batch_size = x.shape[0]
        I = torch.eye(self.n, device=x.device).unsqueeze(0)

        head = self.trunk(x)

        # Generate raw parameters
        q = self.net_q(head)
        b = self.net_bias(head)
        V = self.net_V(head).view(batch_size, self.n, self.n)
        Z = self.net_Z(head).view(batch_size, self.n, self.n)
        Y = self.net_Y(head).view(batch_size, self.n, self.n)

        # Cayley Transform for Skew-Symmetric / Orthogonal components
        P = torch.bmm(Z.transpose(-2, -1), Z) + self.epsilon * I
        L = Y - Y.transpose(-2, -1)
        M = P + L
        S = torch.linalg.solve(I + M, I - M)

        return q, S, V, b

    def _get_A_(self, q, S, V):
        batch_size = q.shape[0]
        device = q.device

        # --- FIX 1: Clamp q to prevent exponential explosion ---
        # If q is > 5, exp(2*q) > 22000. If q > 40, it overflows float32.
        # We clamp it to a safe range.
        q = torch.clamp(q, min=1e-5)

        VtV = V.transpose(-2, -1) @ V + self.epsilon * torch.eye(self.n, device=device, dtype=self.dtype)
        VtV_sq = VtV @ VtV

        exp_d = q
        exp_2d = torch.square(q)

        coeff = 2 * np.sqrt(1 - self.c)

        # Term 1: 2 * sqrt(1 - c) * diag(e^d) * S * (V^T V)
        term1 = coeff * (exp_d.unsqueeze(-1) * (S @ VtV))

        # Term 2: diag(e^{2d}) * (V^T V)^2
        term2 = exp_2d.unsqueeze(-1) * VtV_sq

        return term1 - term2

    def forward(self, x):
        q, S, V, b = self._get_params(x)
        A = self._get_A_(q, S, V)

        # Debugging check (optional, can be removed for speed)
        if torch.isnan(A).any() or torch.isinf(A).any():
            print("Warning: NaN/Inf detected in Matrix A")
            # Fallback to zeros to prevent crash, though training is likely broken if this hits
            A = torch.zeros_like(A)

        w_init = None
        w = self.implicit_cell(A, b, w_init)

        w = self.activation(w)
        out = self.output_layer(w)
        return out