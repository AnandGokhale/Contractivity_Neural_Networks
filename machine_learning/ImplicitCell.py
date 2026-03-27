import torch
import torch.nn as nn


def pr_operator(x, D11, bias, M_inv_t, activation):
            
            z = activation(x)
            u_half = 2*z - x
            rhs = (u_half +  bias).unsqueeze(1)
            z_half = torch.matmul(rhs, M_inv_t).squeeze(1)
            return 2 * z_half - u_half


fast_pr =(pr_operator)


class RENImplicitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D11, bias, alpha, x_init, M_inv_t, activation, max_iter, tol):
        with torch.no_grad():
            x_prev = x_init if x_init is not None else bias
            alpha_bias = alpha * bias
            for _ in range(max_iter):
                x = fast_pr(x_prev, D11, alpha_bias, M_inv_t, activation)
                if torch.norm(x - x_prev) < tol:
                    break
                x_prev = x
            
            z_star = activation(x)
        
        # Save tensors for the analytical backward pass
        ctx.save_for_backward(D11, bias, z_star)
        ctx.activation = activation
        return x 

    @staticmethod
    def backward(ctx, grad_z):
        D11, bias, z_star = ctx.saved_tensors
        activation = ctx.activation
        
        # 1. Compute the local derivative of the activation (phi')
        # Since z = activation(v), for tanh: phi'(v) = 1 - z^2
        if activation == torch.tanh:
            j = 1 - z_star**2
        elif activation == torch.relu:
            # Note: technically we need the pre-activation 'v' here
            # But for ReLU, if z > 0, j = 1, else 0.
            j = (z_star > 0).to(z_star.dtype)
        else:
            # Generic fallback if you use other activations
            with torch.enable_grad():
                z_detached = z_star.detach().requires_grad_(True)
                v_eval = z_detached @ D11.t() + bias
                z_eval = activation(v_eval)
                j = torch.autograd.grad(z_eval, v_eval, grad_outputs=torch.ones_like(z_eval))[0]

        # 2. Solve the Adjoint System (Matching Julia: gn = (I - JW)' \ grad_z)
        batch_size, nv = grad_z.shape
        I = torch.eye(nv, device=D11.device, dtype=D11.dtype).unsqueeze(0)

        J_batch = j.unsqueeze(2) * D11
        A_T_batch = (I - J_batch).transpose(1, 2)
        
        # This solves the Implicit Function Theorem adjoint
        gn = torch.linalg.solve(A_T_batch, grad_z.unsqueeze(2)).squeeze(2)
        grad_v = gn * j 

        grad_D11 = torch.bmm(grad_v.unsqueeze(2), z_star.unsqueeze(1))

        grad_bias = grad_v # Shape [batch, nv] matches input 'bias'

        return grad_D11, grad_bias, None, None, None, None, None, None    


class ImplicitCell(nn.Module):
    def __init__(self, dim: int, activation=torch.tanh, solver_params:dict = {}, device = "cuda"):
        super().__init__()
        self.dim = dim
        self.activation = activation
        self.register_buffer('I', torch.eye(dim))

        if solver_params is None:
            solver_params = {}
        
        self.alpha = solver_params.get("SOLVER_ALPHA", 0.5)
        self.max_iter = solver_params.get("SOLVER_MAX_ITER", 100)
        self.tol = solver_params.get("SOLVER_TOL", 1e-4)
        

    

    def forward(self, D11, bias, x_init, **kwargs):


        M_inv_t = self._compute_M_(D11)
        # Just a clean wrapper for the custom autograd function

        return RENImplicitFunction.apply(
            D11, bias, self.alpha, x_init, M_inv_t, self.activation, self.max_iter, self.tol
        )


    def _compute_M_(self, D11):
        # D11 shape: (Batch, nv, nv)
        batch_size = D11.shape[0]
        
        I_batch = self.I.unsqueeze(0).expand(batch_size, -1, -1)
        
        M = (1 + self.alpha) * I_batch - self.alpha * D11

        M_inv = torch.linalg.inv(M)
        
        # Transpose the last two dimensions for each batch element
        M_inv_t = M_inv.transpose(-2, -1).contiguous()

        return M_inv_t

