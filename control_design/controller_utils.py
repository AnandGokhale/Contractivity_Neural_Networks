import numpy as np
import cvxpy as cp

import torch
import torch.nn as nn

from neuromancer.psl.nonautonomous import TwoTank


sys = TwoTank()


nx = sys.nx 
nu = sys.nu
ny = 1
N_PLANT = 8

N_CONTROLLER = 1 * N_PLANT



class FRNN_NumPy:
    def __init__(self, W, B, C, act_fn='relu'):
        self.W = W
        self.B = B
        self.C = C
        self.act_fn = act_fn

    def forward_state(self, x, u):
        # Calculates dx/dt based on standard Neural ODE structure
        z = self.W @ x + self.B @ u
        if self.act_fn == 'relu':
            activation = np.maximum(0, z)
        else:
            activation = np.tanh(z)
            
        return -x + activation  # dx/dt

    def get_output(self, x):
        # y = Cx
        return self.C @ x
    
    def set_K(self, K):
        self.K = K

def generate_synaptic_weights_stab(W_s, B_s, C_s, c = 0.1):

    n_r = N_CONTROLLER
    n_s = N_PLANT
    n_in = ny
    n_out = nu

    n_total = n_s + n_r
    
    q1_vec = cp.Variable(n_r, nonneg=True)
    q2_vec = cp.Variable(n_s, nonneg=True)
    Q1 = cp.diag(q1_vec)
    Q2 = cp.diag(q2_vec)
    
    # Fully positive definite P
    P = cp.Variable((n_total, n_total), PSD=True)

    Wr_prime = cp.Variable((n_r, n_r))
    Br_prime = cp.Variable((n_r, n_in))

    C_r = np.eye(n_out, n_r)  # Simple output layer to map controller state to control input

    row1 = cp.hstack([Wr_prime, -Br_prime @ C_s])
    row2 = cp.hstack([Q2 @ B_s @ C_r, Q2 @ W_s])

    QW_cl = cp.vstack([row1, row2])

    block11 = -2 * (1 - c) * P
    block22 = -2 * cp.bmat([
        [Q1, np.zeros((n_r, n_s))],
        [np.zeros((n_s, n_r)), Q2]
    ])
    block12 = P + QW_cl.T
    
    lmi = cp.bmat([
        [block11, block12],
        [block12.T, block22]
    ])
    
    constraints = [lmi << -1e-6 * np.eye(2 * n_total)]
    # Add a constraint to keep P and Q from vanishing
    constraints += [cp.trace(P) >= 1]
    
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS) # Or cp.MOSEK if available
    
    if prob.status not in ["optimal", "feasible"]:
        raise ValueError("LMI not feasible for given c")

    Q1_val = Q1.value
    W_r = np.linalg.inv(Q1_val) @ Wr_prime.value
    B_r = np.linalg.inv(Q1_val) @ Br_prime.value

    return W_r, B_r, C_r, P.value

