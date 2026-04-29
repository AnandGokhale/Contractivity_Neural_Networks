import torch
import random

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators
from neuromancer.psl.signals import sines, step, arma, spline
from neuromancer.psl.nonautonomous import TwoTank

import cvxpy as cp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

from FRNN import FRNN
from controller_utils import FRNN_NumPy, generate_synaptic_weights_stab


SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # Required if you are using multi-GPU

# used for reference generation
my_seeded_rng = np.random.default_rng(seed=100)



K_P = 0
K_I = 25

def norm(x):
    return sys.normalize(x, key='X')

def denorm(u):
    return sys.denormalize(u, key='U')


def design_K(FRNN_s, c_r=0.01, delta=0.1):

    W = FRNN_s.W
    B = FRNN_s.B
    C = FRNN_s.C



    n = W.shape[0]
    m = B.shape[1]
    p = C.shape[0]


    A = np.eye(n) - W

    P = cp.Variable((m, m), symmetric=True)
    Y = cp.Variable((m, p))
    q = cp.Variable(n, nonneg=True)          # Q = diag(q), diagonal PD

    Q  = cp.diag(q)

    R = 2*delta * (A.T @ Q @ A) + (1-delta) * (Q @ A + A.T @ Q)
    Z = B.T @ Q @ ((1-delta)*np.eye(n) + 2*delta*A)

    block11 = 2*c_r*P - 2*delta*(B.T @ Q @ B)
    block12 = Z - Y @ C
    block22 = -R



    lmi = cp.bmat([
        [block11,       block12],
        [(block12).T,   block22]
    ])
    prob = cp.Problem(cp.Minimize(0), [lmi << 0, P >> 1e-4*np.eye(m)])
    prob.solve(solver=cp.SCS)
    return np.linalg.solve(P.value, Y.value)



def system_integral_learnt(t, state):
    xs = state[0:N_PLANT].reshape(-1, 1)                  # Plant latent state
    u_int = state[N_PLANT:].reshape(-1, 1)            # Integrator state


    k_i = K_I * np.eye(nu)
    k_p = K_P * np.eye(nu)
    
    r = get_reference(t)

    y_s = FRNN_s.get_output(xs)  # Plant physical output

    # 2. State derivatives
    x_s_dot = FRNN_s.forward_state(xs, u_int) 

    u_int_dot = k_i @ FRNN_s.K @ (r - y_s) 

    return np.concatenate([x_s_dot.flatten(), u_int_dot.flatten()])


def system_integral_new(t, state, sys):

    x_s = state[0:nx].reshape(-1, 1)                  # Plant physical state
    u_int = state[nx:].reshape(-1, 1)            # Integrator state  

    k_i = K_I * np.eye(nu)
    k_p = K_P * np.eye(nu)
    
    r = get_reference(t)

    # 2. State derivatives

    u_int_denorm = denorm(u_int.flatten())
    x_s_dot = sys.equations(t, x_s.flatten(), u_int_denorm.flatten())  # True plant dynamics for reference

    y_s_norm = norm(x_s.flatten())[:ny].reshape(-1, 1)
    
    u_int_dot = k_i @  FRNN_s.K @ (r - y_s_norm) 

    return np.concatenate([x_s_dot.flatten(), u_int_dot.flatten()])


sys = TwoTank()


# 1. Load the trained weights into the same architecture
nx = sys.nx 
nu = sys.nu

ny = 1
N_PLANT = 8

N_CONTROLLER = 1 * N_PLANT
T_sim = 1000


plant = FRNN(nx_orig=nx, nu=nu, nx_ext=N_PLANT, nonlin=torch.nn.Tanh)
plant.load_state_dict(torch.load('trained_sysid_dx_two_tank.pth'))
plant.eval()

# W_s = plant.A.weight.detach().numpy()
W_s = plant._get_W().detach().numpy()
B_s = plant.B.weight.detach().numpy()
C_s = plant.C.weight.detach().numpy()

C_s = C_s[1:ny+ 1, :N_PLANT]  

# True system:
FRNN_s = FRNN_NumPy(W_s, B_s, C_s, act_fn='tanh')



K = design_K(FRNN_s)

FRNN_s.set_K(K)

# Generate reference trajectory

nsim = 1000

ref_ori = step(nsim=nsim, d=nx, min=sys.stats['X']['min'], max=sys.stats['X']['max'], rng=my_seeded_rng)
ref = sys.normalize({'X': ref_ori})['X'][:,:ny].reshape(-1, ny)


def get_reference(t):
    """Returns the Zero-Order Hold normalized reference at continuous time t."""
    return ref[int(np.floor(t/5))].reshape(-1, 1)




t_span = (0, 999)
t_eval_points = np.linspace(t_span[0], t_span[1], num=100)

x0_learned = np.zeros(N_PLANT + nu)
x0_true = np.zeros(nx + nu)  


print("Simulating closed-loop system with learned plant...")
sol_learned = solve_ivp(
        lambda t, x: system_integral_learnt(t, x), 
        t_span, 
        x0_learned,
        t_eval = t_eval_points
        )
y_learned = FRNN_s.C @ sol_learned.y[0:N_PLANT, :]  # Shape: (ny, time_steps)

print("Simulating closed-loop system with true plant...")
sol_true = solve_ivp(
        lambda t, x: system_integral_new(t, x, sys), 
        t_span, 
        x0_true,
        t_eval = t_eval_points
        )

x_true = sol_true.y[:nx,:]  # Shape: (nx, time_steps)
y_true = np.zeros((ny, len(x_true[0, :])))
for i in range(len(x_true[0, :])):
    y_true[:, i] = norm(x_true[:, i])[:ny]

t_eval_true = sol_true.t
t_eval_learned= sol_learned.t
r_vals = np.array([get_reference(t).flatten() for t in t_eval_true])


import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,          # IMPORTANT
    "font.family": "serif",
    "mathtext.fontset": "cm",      # Computer Modern (LaTeX-like)
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
})

# normalize the reference values for plotting
if(ny == 1):
    fig, ax = plt.subplots(figsize=(6, 3)) 

    # Plot the 0th index for both reference and output
    # ax.plot(t_eval_true, r_vals[:, 0], 'k--', label='Reference')
    # ax.plot(t_eval_learned, y_learned[0, :], label='Learned plant output')
    # ax.plot(t_eval_true, y_true[0, :], label='True plant output')

    ax.plot(t_eval_true, r_vals[:,0], 'k--', label='Reference')
    ax.plot(t_eval_learned, y_learned[0,:], label='Learned plant')
    ax.plot(t_eval_true, y_true[0,:], label='True plant')

    ax.set_ylabel('Output')
    ax.set_xlabel('Time [s]')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

else:
    # Plotting
    fig, ax = plt.subplots(nrows=ny, figsize=(10, 8), sharex=True)
     
    for i in range(ny):
        ax[i].plot(t_eval_true, r_vals[:, i], 'k--', label=f'Reference {i}')
        ax[i].plot(t_eval_learned, y_learned[i, :], label=f'Learned plant output {i}')
        ax[i].plot(t_eval_true, y_true[i, :], label=f'True plant output {i}')
        ax[i].set_ylabel(f'y[{i}]')
        ax[i].legend(loc='upper right')
        ax[i].grid(True)
        
    ax[ny - 1].set_xlabel("Time [s]")

plt.tight_layout()

plt.savefig('controller_tracking_new.pdf', bbox_inches='tight')
