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
        discrete_sys_size[ii] = math.floor((X_range[ii, 1] - X_range[ii, 0] - eta_x[ii]) / eta_x[ii] + 1)
    for ii in range(dim_x, dim_x + dim_u):
        discrete_sys_size[ii] = math.floor(
            (U_range[ii - dim_x, 1] - U_range[ii - dim_x, 0] - eta_u[ii - dim_x]) / eta_u[ii - dim_x] + 1)
    return discrete_sys_size.astype(int)


def compute_radius(inp, forw_or_back):
    """Compute the radius of the over-approximated disc"""
    if forw_or_back == 1:
        L = Jacob_forw(inp)
    else:
        L = Jacob_back(inp)

    f = lambda t: linalg.expm(L * t)
    tv = np.linspace(0, sample_time, nbins)
    z = np.apply_along_axis(f, 0, tv.reshape(1, -1))
    matrix_exp_int = np.zeros((dim_x, dim_x))
    for i in range(nbins - 1):
        matrix_exp_int += .5 * np.squeeze(z[:, :, i] + z[:, :, i + 1]) * sample_time / nbins
    #  matrix_exp_int = np.squeeze(z[:, :, nbins - 1]) #* 1.01
    r = (np.dot(linalg.expm(L*sample_time), eta_x) * .5 + np.dot(matrix_exp_int, W)) #* 1.01
    return np.squeeze(r)


def generate_TS_training_data(dynamics, x_tr, y_tr, FW_or_BW):
    """This function is invoked when training data are needed to be written on Disk"""
    #dynamics_coded = ray.put(dynamics)
    for step in range(num_state_inp_pairs // num_tasks_per_step+1):
        num_tasks_in_this_step = min(num_tasks_per_step, num_state_inp_pairs-step*num_tasks_per_step)
        #startt = time.time()
        input_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                  num_tasks_in_this_step-i*length, FW_or_BW)[0]
                                                  for i in range(0, num_tasks_in_this_step // length+1)]
        output_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                     num_tasks_in_this_step - i * length, FW_or_BW)[1]
                    for i in range(0, num_tasks_in_this_step // length + 1)]
        input_decoded = ray.get(input_coded)
        output_decoded = ray.get(output_coded)

        del input_coded, output_coded
        for i in range(num_tasks_in_this_step // length + 1):
            length_for_this_step = min(length, num_tasks_in_this_step - i * length)
            for j in range(length_for_this_step):
                x_tr[step * num_tasks_per_step + i * length + j, :] = input_decoded[i][j]
                y_tr[step * num_tasks_per_step + i * length + j, :] = output_decoded[i][j]
        del input_decoded, output_decoded
        #print("one round takes", time.time()-startt)
    print('GB computation is finished', time.time() - start)


@ray.remote
class ListActor:
    def __init__(self, d1, d2, d3):
        self.inp_ds = d1
        self.out_ds_c = d2
        self.out_ds_r = d3

    def get_inp_ds(self):
        return self.inp_ds

    def get_out_ds_c(self):
        return self.out_ds_c

    def get_out_ds_r(self):
        return self.out_ds_r

    def set_inp_ds(self, i, val):
        self.inp_ds[i, :] = val[:]

    def set_out_ds_c(self, i, val):
        self.out_ds_c[i, :] = val[:]

    def set_out_ds_r(self, i, val):
        self.out_ds_r[i, :] = val[:]


@ray.remote(num_returns=2)
def single_data_generation_TS(ii, dynamics, num_tasks, FW_or_BW):
    length_for_this_step = min(length, num_tasks)
    xx = []
    yy = []
    sample_input = np.zeros(dim_x+dim_u)  # initiating the array containing an input
    for k in range(length_for_this_step):
        pre_j = int(ii*length+k)
        for j in reversed(range(dim_x + dim_u)):
            if j == 0:
                ind_j = pre_j
            else:
                ind_j = pre_j // np.prod(discrete_sys_size[0:j])
                pre_j -= ind_j * np.prod(discrete_sys_size[0:j])
            if j < dim_x:
                sample_input[j] = X_range[j, 0] + rr_x[j] + eta_x[j] * ind_j
            else:
                sample_input[j] = U_range[j - dim_x, 0] + rr_u[j - dim_x] + eta_u[j - dim_x] * ind_j
        xx.append(copy.copy(sample_input))
        inpp = np.transpose(sample_input[dim_x:dim_x + dim_u])
        center = np.transpose(sample_input[0:dim_x])
        nom_next = integrate.odeint(dynamics, center, time_span, args=(inpp,))[nbins - 1, :]
        sample_next1 = nom_next + compute_radius(inpp, FW_or_BW)  # upper right corner
        sample_next2 = nom_next - compute_radius(inpp, FW_or_BW)  # lower left corner
        ind_next1 = ((sample_next1 - X_range[:, 0]) // np.squeeze(eta_x)).astype(int)
        ind_next2 = ((sample_next2 - X_range[:, 0]) // np.squeeze(eta_x)).astype(int)
        y = prepare_training_data(ind_next1, ind_next2)
        yy.append(y)
    return [xx, yy]


def prepare_training_data(ub, lb):
    y = np.zeros(nn)
    start_id = 0
    for dim in range(dim_x):
        y[start_id + shift_no + ub[dim]] = 1
        y[start_id + discrete_sys_size[dim] + (1+2+1) * shift_no + lb[dim]] = 1
        start_id = 2*(np.sum(discrete_sys_size[0:dim+1])+3*(dim+1)*shift_no)
        #start_id = (np.sum(discrete_sys_size[0:dim + 1]) + 2 * (dim + 1) * shift_no)
    return y


# Listing the example-specific parameters
env = env_pool.tora()  # The selected case-study

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
eta_x = env.eta_x  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
shift_no = env.shift_no  # shifting the index vectors to avoid negative values
W = env.W  # disturbance bounds
nbins = env.nbins  # used for computing the integral
time_span = env.time_span  # used for computing the integral
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# defining filenames for saving the transition system
forw_inp_TS_filename = env.forw_inp_TS_filename
forw_out_TS_filename = env.forw_out_TS_filename
back_inp_TS_filename = env.back_inp_TS_filename
back_out_TS_filename = env.back_out_TS_filename
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
nn = 2 * (np.sum(discrete_sys_size[0:dim_x])+3*shift_no*dim_x)  # dimension of the vector at the output of the trained NN
# Only for the systems whose main diagonal of L doesn't depend on control input:
#  U_range_non_sym = np.maximum(np.zeros(dim_u, 2), U_range) # Don't uncomment!!!

# Generate training data for the forward transition system
#dynamics = dynamics_forw
forw_inp_ds = np.memmap(forw_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
forw_out_ds = np.memmap(forw_out_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, nn), offset=0)

back_inp_ds = np.memmap(back_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
back_out_ds = np.memmap(back_out_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, nn), offset=0)

# Generate training data with writing over disk
ray.init(_plasma_directory="/tmp", log_to_driver=False)
start = time.time()

"""inp_ds_id = ray.put(inp_ds)
out_ds_c_id = ray.put(out_ds_c)
out_ds_r_id = ray.put(out_ds_r)"""

#Var_set = ListActor.remote(inp_ds, out_ds_c, out_ds_r)

generate_TS_training_data(dynamics_forw, forw_inp_ds, forw_out_ds, 1)
generate_TS_training_data(dynamics_back, back_inp_ds, back_out_ds, -1)
ray.shutdown()
forw_inp_ds.flush()
forw_out_ds.flush()
back_inp_ds.flush()
back_out_ds.flush()
print("Execution time for writing on disk is", time.time() - start)
