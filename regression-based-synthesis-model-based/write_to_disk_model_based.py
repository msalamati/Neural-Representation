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


"""def generate_TS_training_data(dynamics, x_tr, y_tr_c, y_tr_r1, y_tr_r2, FW_or_BW):

    #dynamics_coded = ray.put(dynamics)
    for step in range(num_state_inp_pairs // num_tasks_per_step+1):
        num_tasks_in_this_step = min(num_tasks_per_step, num_state_inp_pairs-step*num_tasks_per_step)
        #startt = time.time()
        input_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                  num_tasks_in_this_step-i*length, FW_or_BW)[0]
                                                  for i in range(0, num_tasks_in_this_step // length+1)]
        center_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length, FW_or_BW)[1]
                                                        for i in range(0, num_tasks_in_this_step // length + 1)]
        radius1_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length, FW_or_BW)[2]
                                                        for i in range(0, num_tasks_in_this_step // length + 1)]
        radius2_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                     num_tasks_in_this_step - i * length, FW_or_BW)[3]
                           for i in range(0, num_tasks_in_this_step // length + 1)]
        input_decoded = ray.get(input_coded)
        center_decoded = ray.get(center_coded)
        radius1_decoded = ray.get(radius1_coded)
        radius2_decoded = ray.get(radius2_coded)

        #[single_data_generation_TS(step * num_tasks_per_step // length + i, dynamics,
                               #   num_tasks_in_this_step-i*length, inp_ds, out_ds_c, out_ds_r) for i in range(0, num_tasks_in_this_step // length+1)]
        del input_coded, center_coded, radius1_coded, radius2_coded
        for i in range(num_tasks_in_this_step // length + 1):
            length_for_this_step = min(length, num_tasks_in_this_step - i * length)
            for j in range(length_for_this_step):
                x_tr[step * num_tasks_per_step + i * length + j, :] = input_decoded[i][j]
                y_tr_c[step * num_tasks_per_step + i * length + j, :] = center_decoded[i][j]
                y_tr_r1[step * num_tasks_per_step + i * length + j, :] = radius1_decoded[i][j]
                y_tr_r2[step * num_tasks_per_step + i * length + j, :] = radius2_decoded[i][j]
        del input_decoded, center_decoded, radius1_decoded, radius2_decoded
        #print("one round takes", time.time()-startt)
    print('GB computation is finished', time.time() - start)"""


def generate_TS_training_data(dynamics, inp_ds, out_ds_c, out_ds_r, FW_or_BW):
    """This function is invoked when training data are needed to be written on Disk"""
    for step in range(num_state_inp_pairs // num_tasks_per_step+1):
        num_tasks_in_this_step = min(num_tasks_per_step, num_state_inp_pairs-step*num_tasks_per_step)
        input_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                  num_tasks_in_this_step-i*length, FW_or_BW)[0]
                                                  for i in range(0, num_tasks_in_this_step // length+1)]
        center_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length, FW_or_BW)[1]
                                                        for i in range(0, num_tasks_in_this_step // length + 1)]
        radius_coded = [single_data_generation_TS.remote(step * num_tasks_per_step // length + i, dynamics,
                                                        num_tasks_in_this_step - i * length, FW_or_BW)[2]
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
def single_data_generation_TS(ii, dynamics, num_tasks, FW_or_BW):
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
                sample_input[j] = X_range[j, 0] + rr_x[j] + eta_x[j] * ind_j
            else:
                sample_input[j] = U_range[j - dim_x, 0] + rr_u[j - dim_x] + eta_u[j - dim_x] * ind_j
        inpp = np.transpose(sample_input[dim_x:dim_x + dim_u])
        center = np.transpose(sample_input[0:dim_x])
        nom_next = integrate.odeint(dynamics, center, time_span, args=(inpp,))[nbins - 1, :]
        xx.append(copy.copy(sample_input))
        cc.append(copy.copy(nom_next))
        r = compute_radius(inpp, FW_or_BW)
        rr.append(copy.copy(r))
    return [xx, cc, rr]


""""@ray.remote(num_returns=4)
def single_data_generation_TS(ii, dynamics, num_tasks, FW_or_BW):
    length_for_this_step = min(length, num_tasks)
    xx = []
    cc = []
    rr1 = []
    rr2 = []
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
        inpp = np.transpose(sample_input[dim_x:dim_x + dim_u])
        center = np.transpose(sample_input[0:dim_x])
        nom_next = integrate.odeint(dynamics, center, time_span, args=(inpp,))[nbins - 1, :]
        xx.append(copy.copy(sample_input))
        sample_next1 = nom_next + compute_radius(inpp, FW_or_BW)  # upper right corner
        sample_next2 = nom_next - compute_radius(inpp, FW_or_BW)  # lower left corner
        ind_nom = ((nom_next - X_range[:, 0]) // np.squeeze(eta_x)).astype(int)
        ind_next1 = ((sample_next1 - X_range[:, 0]) // np.squeeze(eta_x)).astype(int)
        ind_next2 = ((sample_next2 - X_range[:, 0]) // np.squeeze(eta_x)).astype(int)
        border_next1 = np.zeros(dim_x)
        for dim in range(dim_x):
            if ind_nom[dim] == ind_next1[dim]:
                if ind_nom[dim] == ind_next2[dim]:
                    nom_next[dim] = X_range[dim, 0] + ind_nom[dim] * np.squeeze(eta_x)[dim] + .5 * np.squeeze(eta_x)[dim]
                    border_next1[dim] = nom_next[dim] + alpha * np.squeeze(eta_x)[dim]
                else:
                    nom_next[dim] = X_range[dim, 0] + ind_nom[dim] * np.squeeze(eta_x)[dim] + (alpha/2) * np.squeeze(eta_x)[dim]
                    border_next1[dim] = nom_next[dim] + (alpha/2) * np.squeeze(eta_x)[dim]
                    #  border_next1[dim] = nom_next[dim] + alpha * np.squeeze(eta_x)[dim]  #sample_next1[dim]
            else:
                border_next1[dim] = X_range[dim, 0] + ind_next1[dim] * np.squeeze(eta_x)[dim] + alpha * np.squeeze(eta_x)[dim]

        border_next2 = np.zeros(dim_x)
        for dim in range(dim_x):
            if ind_nom[dim] == ind_next2[dim]:
                if ind_nom[dim] == ind_next1[dim]:
                    border_next2[dim] = nom_next[dim] - alpha * np.squeeze(eta_x)[dim]
                else:
                    nom_next[dim] = X_range[dim, 0] + ind_nom[dim] * np.squeeze(eta_x)[dim] + (1-alpha/2) * np.squeeze(eta_x)[dim]
                    border_next2[dim] = nom_next[dim] - (alpha / 2) * np.squeeze(eta_x)[dim]
            else:
                border_next2[dim] = X_range[dim, 0] + ind_next2[dim] * np.squeeze(eta_x)[dim] + (1-alpha) * np.squeeze(eta_x)[dim]

        #border_next1 = X_range[:, 0] + ind_next1 * np.squeeze(eta_x)+alpha*np.squeeze(eta_x)
        #border_next2 = X_range[:, 0] + ind_next2 * np.squeeze(eta_x)+(1-alpha)*np.squeeze(eta_x)
        r_modified1 = abs(border_next1 - nom_next)
        r_modified2 = abs(border_next2 - nom_next)
        cc.append(copy.copy(nom_next))
        rr1.append(copy.copy(r_modified1))
        rr2.append(copy.copy(r_modified2))
    #print("time per pair is", time.time()-startt)
    return [xx, cc, rr1, rr2]"""




# Listing the example-specific parameters
env = env_pool.unicycle()  # The selected case-study

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
eta_x = env.eta_x  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
alpha = 0.1  # 1-alpha*eta_x is the value which will be deducted from the actual radius that is considered for the training phase (The final ove-approximated set of cells would be the same)
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
discrete_sys_size_scaled = 2*discrete_sys_size
discrete_inp_size = discrete_sys_size[dim_x:dim_x+dim_u]  # vector containing number of discrete points along each dimension in the input space
# state and input spaces
num_dis_states = np.prod(discrete_sys_size[0:dim_x]).astype(int)  # size of the state-space
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
num_state_inp_pairs = np.prod(discrete_sys_size).astype(int)  # number of state-input pairs
# Only for the systems whose main diagonal of L doesn't depend on control input:
#  U_range_non_sym = np.maximum(np.zeros(dim_u, 2), U_range) # Don't uncomment!!!

# Generate training data for the forward transition system
#dynamics = dynamics_forw
inp_ds_FW = np.memmap(forw_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
out_ds_c_FW = np.memmap(forw_out_TS_c_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)
out_ds_r_FW = np.memmap(forw_out_TS_r_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)

inp_ds_BW = np.memmap(back_inp_TS_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x + dim_u), offset=0)
out_ds_c_BW = np.memmap(back_out_TS_c_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)
out_ds_r_BW = np.memmap(back_out_TS_r_filename, dtype='float32', mode='w+', shape=(num_state_inp_pairs, dim_x), offset=0)

# Generate training data with writing over disk
ray.init(_plasma_directory="/tmp", log_to_driver=False)
start = time.time()

# Creating the ray object stores corresponding to different datasets
inp_ds_id_FW = ray.put(inp_ds_FW)
out_ds_c_id_FW = ray.put(out_ds_c_FW)
out_ds_r_id_FW = ray.put(out_ds_r_FW)
inp_ds_id_BW = ray.put(inp_ds_BW)
out_ds_c_id_BW = ray.put(out_ds_c_BW)
out_ds_r_id_BW = ray.put(out_ds_r_BW)

#Var_set = ListActor.remote(inp_ds, out_ds_c, out_ds_r)

generate_TS_training_data(dynamics_forw, inp_ds_FW, out_ds_c_FW, out_ds_r_FW, 1)
inp_ds_FW.flush()
out_ds_c_FW.flush()
out_ds_r_FW.flush()

generate_TS_training_data(dynamics_back, inp_ds_BW, out_ds_c_BW, out_ds_r_BW, -1)
inp_ds_BW.flush()
out_ds_c_BW.flush()
out_ds_r_BW.flush()

ray.shutdown()

print("Execution time for writing on disk is", time.time() - start)
