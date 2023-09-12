import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
from casadi import *
# Import do_mpc package:
import do_mpc

# states: 7 joint angles
theta = model.set_variable(var_type='_x', var_name='theta', shape=(7,1))

# control: changle of angles
dtheta = model.set_variable(var_type='_u', var_name='dtheta', shape=(7,1))

model.set_rhs('theta', theta + dtheta)

model.setup()

cost = np.random(1)

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)