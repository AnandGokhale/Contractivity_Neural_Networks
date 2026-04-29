from neuromancer.psl.nonautonomous import TwoTank
from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.trainer import Trainer


import torch
import torch.optim as optim
# create dataloaders
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from FRNN import FRNN, StateLifter

N_EXT = 8



sys = TwoTank()


nx = sys.nx
nu = sys.nu
ny = sys.ny

# Pre-lift initial condition in data prep
lifter = StateLifter(nx_orig=sys.nx, nx_ext=N_EXT, learned=False)  # or learned=True
dx = FRNN(nx_orig=sys.nx, nu=sys.nu, nx_ext=N_EXT, nonlin=torch.nn.Tanh)



train_data, dev_data, test_data = [sys.simulate(nsim=1000) for i in range(3)]
train_data, dev_data, test_data = [sys.normalize(d) for d in [train_data, dev_data, test_data]]


for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(100, 10, nx)
    d['U'] = d['U'].reshape(100, 10, nu)
    d['Y'] = d['Y'].reshape(100, 10, ny)

    d['x0'] = d['X'][:, 0:1, :] 
    d['Time'] = d['Time'].reshape(100, -1)



train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader = [DataLoader(d, batch_size=100, collate_fn=d.collate_fn, shuffle=True)
                            for d in [train_dataset, dev_dataset]]



# define neural ODE (NODE)
# dx = blocks.MLP(sys.nx + sys.nu, sys.nx, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.Tanh,
#               hsizes=[20 for h in range(3)])


interp_u = lambda tq, t, u: u
# we use ODE integrators to discretize continuous-time NODE model
integrator = integrators.Euler(dx, h=torch.tensor(0.1), interp_u=interp_u)
# model.show()

# encoder_node = Node(dx.encoder, ['x0'], ['xn'])
encoder_node = Node(dx.encoder, ['x0'], ['xn'], name='encoder')

ode_node     = Node(integrator, ['xn', 'U'], ['xn'], name='ode_integrator')
decoder_node = Node(dx.C, ['xn'], ['xpred'], name='decoder')          

# 3. Name the System as well
model = System([ode_node, decoder_node], name='recurrent_model')



xpred = variable('xpred')[:, :, :]
# Nstep rollout predictions from the model
#xpred = variable('xn')[:, :-1, :]
# Ground truth data
xtrue = variable('X')
# define system identification loss function
loss = (xpred == xtrue) ^ 2
loss.update_name('loss')

# sample = next(iter(train_loader))
# with torch.no_grad():
#     out = model(sample)
# print('xn shape:    ', out['xn'].shape)
# print('xpred shape: ', out['xpred'].shape)
# print('X shape:     ', sample['X'].shape)
# print('U shape:     ', sample['U'].shape)

# construct differentiable optimization problem in Neuromancer
obj = PenaltyLoss([loss], [])
problem = Problem([encoder_node, model], obj)



#opt = optim.Adam(model.parameters(), 0.001)

opt = optim.Adam(list(model.parameters()) + list(encoder_node.parameters()), 0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    opt, 
    mode='min', 
    factor=0.5, 
    patience=50, 
    min_lr=1e-6
)

trainer = Trainer(problem, train_loader, dev_loader, dev_loader,
                  optimizer=opt,
                  epochs=1000,
                  patience=300,
                  train_metric='train_loss',
                  eval_metric='dev_loss',
                  lr_scheduler=scheduler,)
best_model = trainer.train()

torch.save(dx.state_dict(), 'trained_sysid_dx_two_tank.pth')
print("Model weights successfully saved to 'trained_sysid_dx.pth'")



test_data = sys.normalize(sys.simulate(nsim=1000))
test_data['X'] = test_data['X'].reshape(1, *test_data['X'].shape)  # (1, 1000, 3)
test_data['U'] = test_data['U'].reshape(1, *test_data['U'].shape)  # (1, 1000, 3)

# Use encoder to lift x0 -> z0, just like during training
x0 = torch.tensor(test_data['X'][:, 0:1, :], dtype=torch.float32)  # (1, 1, 3)
with torch.no_grad():
    z0 = dx.encoder(x0)   # (1, 1, 60)
test_data['xn'] = z0

test_data = {k: torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v
             for k, v in test_data.items()}

test_output = model(test_data)

fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
for v in range(2):
    ax[v].plot(test_output['xpred'][0, :-1, v].detach().numpy(), label='pred')
    ax[v].plot(test_data['X'][0, :, v].detach().numpy(), '--', label='true')
    ax[v].set_ylabel(f'x[{v}]')
    ax[v].legend()
fig.savefig('sys_id_neuromancer.png', dpi=150, bbox_inches='tight')