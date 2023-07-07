"""By running this code, for each of the forward and backward dynamics, two arrays will be saved into the disk:
 One array contains the set of (discrete) state-input pairs,
  and the other array contains the successor states (for each state-input pair)."""

import os
import numpy as np
import math
import scipy.integrate as integrate
import time
from scipy import linalg
import ray
import copy
import env_pool

import cvxpy as cvx

# Defining the functions
def discrete_sys_size_gen():
    """ This function computes a vector that contains number of
     discrete states for every dimension of state and input spaces."""
    discrete_sys_size = np.zeros(dim_x + dim_u)
    for ii in range(0, dim_x):
        discrete_sys_size[ii] = math.floor((X_range_FW[ii, 1] - X_range_FW[ii, 0] - eta_x[ii]) / eta_x[ii] + 1)
    for ii in range(dim_x, dim_x + dim_u):
        discrete_sys_size[ii] = math.floor(
            (U_range[ii - dim_x, 1] - U_range[ii - dim_x, 0] - eta_u[ii - dim_x]) / eta_u[ii - dim_x] + 1)
    return discrete_sys_size.astype(int)


def generate_TS_training_data(dynamics, inp_ds, out_ds_c, out_ds_r):
    """This function is invoked when training data are needed to be written on Disk"""
    for step in range(num_state_inp_pairs // num_tasks_per_step+1):
        num_tasks_in_this_step = min(num_tasks_per_step, num_state_inp_pairs-step*num_tasks_per_step)
        input_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                  num_tasks_in_this_step-i*length)[0]
                                                  for i in range(0, num_tasks_in_this_step // length+1)]
        center_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length)[1]
                                                        for i in range(0, num_tasks_in_this_step // length + 1)]
        radius_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length)[2]
                                                        for i in range(0, num_tasks_in_this_step // length + 1)]
        input_decoded = ray.get(input_coded)
        center_decoded = ray.get(center_coded)
        radius_decoded = ray.get(radius_coded)
        del input_coded, center_coded, radius_coded
        for i in range(num_tasks_in_this_step // length + 1):
            length_for_this_step = min(length, num_tasks_in_this_step - i * length)
            for j in range(length_for_this_step):
                inp_ds[step * num_tasks_per_step + i * length + j, :] = input_decoded[i][j]
                out_ds_c[step * num_tasks_per_step + i * length + j, :] = center_decoded[i][j]
                out_ds_r[step * num_tasks_per_step + i * length + j, :] = radius_decoded[i][j]
        del input_decoded, center_decoded, radius_decoded


@ray.remote(num_returns=3)
def single_data_generation_TS(ii, dynamics, num_tasks):
    length_for_this_step = min(length, num_tasks)
    xx = []
    cc = []
    rr = []
    sample_input = np.zeros(dim_x+dim_u)  # initiating the array containing an input
    # First compute the cell value corresponding to ind_BW w.r.t X_range_FW
    for k in range(length_for_this_step):
        pre_j = int(ii*length+k)
        for j in reversed(range(dim_x + dim_u)):
            if j == 0:
                ind_j = pre_j
            else:
                ind_j = pre_j // np.prod(discrete_sys_size[0:j])
                pre_j -= ind_j * np.prod(discrete_sys_size[0:j])
            if j < dim_x:
                sample_input[j] = X_range_FW[j, 0] + rr_x[j] + eta_x[j] * ind_j
            else:
                sample_input[j] = U_range[j - dim_x, 0] + rr_u[j - dim_x] + eta_u[j - dim_x] * ind_j
        inpp = np.transpose(sample_input[dim_x:dim_x + dim_u])
        center = np.transpose(sample_input[0:dim_x])
        nom_next = integrate.odeint(dynamics, center, time_span, args=(inpp,))[nbins - 1, :]
        xx.append(copy.copy(sample_input))
        cc.append(copy.copy(nom_next))
        theta2 = cvx.Variable(dim_x)
        constraints = []
        sample_state_list = np.random.uniform(center - np.squeeze(eta_x) / 2, center + np.squeeze(eta_x) / 2, (num_samples, dim_x))
        for samp_idx in range(num_samples):
            sample_state = sample_state_list[samp_idx, :]
            sample_next = integrate.odeint(dynamics, sample_state, time_span, args=(inpp,))[nbins - 1, :]
            constraints += [np.abs(sample_next - nom_next) - theta2 - gamma <= 0, theta2 >= 0]
        c_obj = np.ones((1, dim_x))
        objective = cvx.Minimize(np.squeeze(c_obj) @ theta2)
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        r = theta2.value + gamma
        rr.append(copy.copy(r))
    return [xx, cc, rr]


# Listing the example-specific parameters
env = env_pool.tora()  # here enter the name of the target case study

gamma = env.gamma  # The added bias value
num_samples = env.num_samples  # number of samples to be taken for each state-input pair
X_range = env.X_range  # state-space
X_range_FW = env.X_range_FW  # state-space for the FW computations
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
eta_x = env.eta_x  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
W = env.W  # disturbance bounds
nbins = env.nbins  # used for computing the integral
time_span = env.time_span  # used for computing the integral
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# defining filenames for saving the transition system
forw_inp_TS_filename = env.forw_inp_TS_filename
forw_out_TS_c_filename = env.forw_out_TS_c_filename
forw_out_TS_r_filename = env.forw_out_TS_r_filename
back_inp_TS_filename = env.back_inp_TS_filename
back_out_TS_c_filename = env.back_out_TS_c_filename
back_out_TS_r_filename = env.back_out_TS_r_filename
# define the env related methods
dynamics_forw = env.dynamics_forw
dynamics_back = env.dynamics_back
Jacob_forw = env.Jacob_forw
Jacob_back = env.Jacob_back

# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen()  # vector containing number of discrete points along each dimension in the state-input space
discrete_inp_size = discrete_sys_size[dim_x:dim_x+dim_u]  # vector containing number of discrete points along each dimension in the input space
# state and input spaces
num_dis_states = np.prod(discrete_sys_size[0:dim_x]).astype(int)  # size of the state-space
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
num_state_inp_pairs = np.prod(discrete_sys_size).astype(int)  # number of state-input pairs

# Generate training data for the forward transition system
inp_ds_FW = np.memmap(forw_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
out_ds_c_FW = np.memmap(forw_out_TS_c_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)
out_ds_r_FW = np.memmap(forw_out_TS_r_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)

inp_ds_BW = np.memmap(back_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
out_ds_c_BW = np.memmap(back_out_TS_c_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)
out_ds_r_BW = np.memmap(back_out_TS_r_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)

# Generate training data with writing over disk
ray.init(_plasma_directory="/tmp", log_to_driver=False)  # Initializing ray
start = time.time()

# Creating the ray object stores corresponding to different datasets
inp_ds_id_FW = ray.put(inp_ds_FW)
out_ds_c_id_FW = ray.put(out_ds_c_FW)
out_ds_r_id_FW = ray.put(out_ds_r_FW)
inp_ds_id_BW = ray.put(inp_ds_BW)
out_ds_c_id_BW = ray.put(out_ds_c_BW)
out_ds_r_id_BW = ray.put(out_ds_r_BW)


generate_TS_training_data(dynamics_forw, inp_ds_FW, out_ds_c_FW, out_ds_r_FW)
inp_ds_FW.flush()
out_ds_c_FW.flush()
out_ds_r_FW.flush()

generate_TS_training_data(dynamics_back, inp_ds_BW, out_ds_c_BW, out_ds_r_BW)
inp_ds_BW.flush()
out_ds_c_BW.flush()
out_ds_r_BW.flush()
ray.shutdown()

print("Execution time for writing on disk is", time.time() - start)
