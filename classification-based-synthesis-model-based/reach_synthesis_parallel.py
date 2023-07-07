"""By running this code: 1- winning domain for a given instance of reach-avoid problem is computed and saved into the
 disk; 2- The controller for the computed winning domain is synthesized and saved into the disk 3- in case answer to
  the given instance of problem is positive, a (nominal) trajectory would be generated...As Input, only the example
  specific information should be provided in the 'Setting the parameters' part...The synthesis routine is parallelized
   using the Ray API."""
import os
import numpy as np
import math
from operator import mul
from functools import reduce
import copy
import time
import ray
import scipy.integrate as integrate
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import env_pool



def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__



def NN_structure_TS():
    """Create the model for estimating the transition system."""
    model = Sequential()
    model.add(Dense(40, input_dim=dim_x+dim_u, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(800, activation=tf.nn.relu))
    model.add(Dense(nn, activation='linear'))
    return model



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


def c2d_oapprox(disc, FW_or_BW):
    eps = np.ones(dim_x)  # error margin in terms of number of the cells
    #out_of_box_flag = 0
    if (FW_or_BW != 1) or ((np.floor((disc[:, 0]-X_range[:, 0])/np.squeeze(eta_x))+eps >= np.zeros(disc.shape[0])).all()  and (np.floor((disc[:, 1] - X_range[:, 0]) / np.squeeze(eta_x)) < eps+state_space_size).all()):
        # Compute the set of discrete states associated with a continuous set in the over-approximation sense
        low_vec_discrete = np.squeeze(np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.zeros(dim_x), np.floor((disc[:, 0]-X_range[:, 0])/np.squeeze(eta_x)))))
        high_vec_discrete = np.squeeze(np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.ones(dim_x), np.floor((disc[:, 1]-X_range[:, 0])/np.squeeze(eta_x)))))
        #if min(abs(low_vec_discrete-high_vec_discrete)) == 0:
            #ind_set = []
        #else:
        # The next lines are responsible for generating a list of vectorized indices corresponding to the disc!
        reach_ind_range = np.zeros((dim_x, 2))
        for ind in range(dim_x):
            reach_ind_range[ind, :] = [low_vec_discrete[ind], high_vec_discrete[ind]+1] # Adding 1 is necessary for the next step

        ranges = reach_ind_range.astype(int).tolist()
        operations = reduce(mul, (p[1] - p[0] for p in ranges)) - 1  # No. of reached states - 1
        result = [ii[0] for ii in ranges]  # The first reached state
        pos = len(ranges) - 1  # Set the pointer at the last dimension
        increments = 0
        ind_set = [copy.copy(result)]  # Copying value of the first reached state into ind_set
        while increments < operations:
            if result[pos] == ranges[pos][1] - 1:
                result[pos] = ranges[pos][0]
                pos -= 1  # Set the pointer to the previous dimension
            else:
                result[pos] += 1
                increments += 1
                pos = len(ranges) - 1  # Increment the innermost loop and set the pointer at the last dimension
                ind_set.append(copy.copy(result))
    else:
        ind_set = []

    return ind_set

def c2d_uapprox(disc):
    # Compute the set of discrete states associated with a continuous set in the under-approximation sense
    low_vec_discrete = np.squeeze(np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.zeros(dim_x), np.ceil((disc[:, 0]-X_range[:, 0])/np.squeeze(eta_x)))))
    high_vec_discrete = np.squeeze(np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.ones(dim_x), np.floor((disc[:, 1]-X_range[:, 0])/np.squeeze(eta_x)) - np.ones(dim_x))))
    #if min(abs(low_vec_discrete - high_vec_discrete)) == 0:
        #ind_set = []
    #else:
    reach_ind_range = np.zeros((dim_x, 2))
    for ind in range(dim_x):
        reach_ind_range[ind, :] = [low_vec_discrete[ind], high_vec_discrete[ind]+1] # Adding 1 is necessary for the next step

    ranges = reach_ind_range.astype(int).tolist()
    operations = reduce(mul, (p[1] - p[0] for p in ranges)) - 1  # No. of reached states - 1
    result = [ii[0] for ii in ranges]  # The first reached state
    pos = len(ranges) - 1  # Set the pointer at the last dimension
    increments = 0
    ind_set = [copy.copy(result)]  # Copying value of the first reached state into ind_set
    while increments < operations:
        if result[pos] == ranges[pos][1] - 1:
            result[pos] = ranges[pos][0]
            pos -= 1  # Set the pointer to the previous dimension
        else:
            result[pos] += 1
            increments += 1
            pos = len(ranges) - 1  # Increment the innermost loop and set the pointer at the last dimension
            ind_set.append(copy.copy(result))

    return ind_set



def ind_to_cell(ind, full_range, space_size, eta, dim):
    cell_center = np.zeros(dim)
    rr = eta / 2
    pre_j = ind
    for j in reversed(range(dim)):
        if j == 0:
            ind_j = pre_j
        else:
            ind_j = pre_j // np.prod(space_size[0:j])
            pre_j -= ind_j * np.prod(space_size[0:j])
        cell_center[j] = full_range[j, 0] + rr[j] + eta[j] * ind_j

    return cell_center


def ind_vectorizer(ind, space_size, dim):
    ind_vec = np.zeros(dim)
    pre_j = ind
    for j in reversed(range(dim)):
        if j == 0:
            ind_j = pre_j
        else:
            ind_j = pre_j // np.prod(space_size[0:j])
            pre_j -= ind_j * np.prod(space_size[0:j])
        ind_vec[j] = ind_j
    return ind_vec#.astype(int)


def cell_to_vectorized_ind(state_vec, full_range, eta, space_size, dim):
    ind = np.maximum(np.zeros(dim), np.minimum((state_vec - full_range[:, 0])//np.squeeze(eta), space_size))
    return ind

def vec_ind_to_ind(vec_ind, space_size, dim):
    ind = vec_ind[0]
    for i in range(dim-1):
        ind += vec_ind[i+1]*space_size[i]
    return ind.astype(int)


def update_reachable_set(y):
    """Finding discrete set of states corresponding to the continuous disc (for the analytical solution)"""
    reach_high_vec_discrete_analytical = np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.zeros(dim_x), compute_bounds(y)[0]))  # np.squeeze(np.minimum(tuple(discrete_sys_size[0:dim_x])-np.ones(dim_x), np.maximum(np.zeros(dim_x), np.floor((s2-radii2-err2-X_range[:, 0])/np.squeeze(eta_x)))))
    reach_low_vec_discrete_analytical = np.minimum(state_space_size-np.ones(dim_x), np.maximum(np.zeros(dim_x), compute_bounds(y)[1]))  # np.squeeze(np.minimum(tuple(discrete_sys_size[0:dim_x])-np.ones(dim_x), np.maximum(np.ones(dim_x), np.floor((s2+radii1+err1-X_range[:, 0])/np.squeeze(eta_x)))))
    reach_ind_range = np.zeros((dim_x, 2))
    for ind in range(dim_x):
        reach_ind_range[ind, :] = [reach_low_vec_discrete_analytical[ind], reach_high_vec_discrete_analytical[ind]+1] # Adding 1 is necessary for the next step

    ranges = reach_ind_range.astype(int).tolist()
    operations = reduce(mul, (p[1] - p[0] for p in ranges)) - 1  # No. of reached states - 1
    result = [ii[0] for ii in ranges]  # The first reached state
    pos = len(ranges) - 1  # Set the pointer at the last dimension
    increments = 0
    ind_set = [copy.copy(result)]  # Copying value of the first reached state into ind_set
    while increments < operations:
        if result[pos] == ranges[pos][1] - 1:
            result[pos] = ranges[pos][0]
            pos -= 1  # Set the pointer to the previous dimension
        else:
            result[pos] += 1
            increments += 1
            pos = len(ranges) - 1  # Increment the innermost loop and set the pointer at the last dimension
            ind_set.append(copy.copy(result))

    return ind_set


def compute_bounds(vec):
    ub = np.zeros(dim_x)
    lb = np.zeros(dim_x)
    start_id1 = 0
    for dim in range(dim_x):
        end_id1 = start_id1 + discrete_sys_size[dim] + 3 * shift_no
        ub[dim] = np.argmax(vec[start_id1:end_id1])-shift_no
        start_id2 = end_id1
        end_id2 = start_id2 + discrete_sys_size[dim] + 3 * shift_no
        lb[dim] = np.argmax(vec[start_id2:end_id2])-shift_no
        start_id1 = end_id2  #2*(np.sum(discrete_sys_size[0:dim+1])+2*(dim+1)*shift_no)
    return [ub, lb]

@ray.remote#(num_returns = 1)
def single_state_computation(state, yFW, yBW, controller):
    reach_set_ind_iter = []
    s1 = X_range[:, 0] + np.squeeze(eta_x)/2 + np.squeeze(eta_x) * state
    #startt = time.time()
    for inp_idx in range(num_dis_inputs):
        inp = ind_to_cell(inp_idx, U_range, input_space_size, eta_u, dim_u)
        NN_BW_inp = np.concatenate((s1, inp))
        inp_vec_idx = ind_vectorizer(inp_idx, input_space_size, dim_u)
        pair_idx_BW = np.concatenate((np.array(state), inp_vec_idx))
        #if len(list(set(pair_idx_BW).difference(set(pairs_to_be_stored_list_back_file)))) == 0:
        if pair_idx_BW.tolist() in pairs_to_be_stored_list_back_file.tolist():
            #idx_BW = vec_ind_to_ind(pair_idx_BW, discrete_sys_size, dim_x+dim_u)
            #idx_BW = vec_ind_to_ind(inp_idx, input_space_size, dim_u)
            v_BW = yBW[inp_idx, :]
        else:
            v_BW = np.squeeze(NN_TS_back(np.array([NN_BW_inp])).numpy())
        backward_reachable_tuple = [tuple(item) for item in update_reachable_set(v_BW)]
        reachable_set_tuple = [tuple(item) for item in reach_set_ind]
        candid_pool_ind = list(set(backward_reachable_tuple).difference(set(reachable_set_tuple)))
        for candid in candid_pool_ind:
            ss1 = X_range[:, 0] + np.squeeze(eta_x)/2 + np.squeeze(eta_x) * candid
            NN_FW_inp = np.concatenate((ss1, inp))
            pair_idx_FW = np.concatenate((candid, inp_vec_idx))
            if pair_idx_FW.tolist() in pairs_to_be_stored_list_forw_file.tolist():
                #idx_FW = vec_ind_to_ind(pair_idx_FW, discrete_sys_size, dim_x + dim_u)
                #idx_FW = vec_ind_to_ind(inp_idx, input_space_size, dim_u)
                v_FW = yFW[inp_idx, :]
            else:
                v_FW = np.squeeze(NN_TS_forw(np.array([NN_FW_inp])).numpy())
            forward_reachable_tuple = [tuple(item) for item in update_reachable_set(v_FW)]
            #print(len(list(forward_reachable_tuple)))
            if list(set(forward_reachable_tuple).difference(set(reachable_set_tuple))) == [] and list(set(forward_reachable_tuple)) != []:
                # print("YESSSS")
                reach_set_ind_iter.append(list(candid))
                controller[candid] = inp_idx
    # print(time.time()-startt)
    return reach_set_ind_iter



# Setting the parameters
env = env_pool.tora()

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
Goal_range = env.Goal_range  # target range
Obs_range = env.Obs_range  # obstacle range
eta_x = env.eta_x  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
shift_no = env.shift_no
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# defining filenames for saving the transition system
forw_inp_TS_filename = env.forw_inp_TS_filename
forw_out_TS_filename = env.forw_out_TS_filename
back_inp_TS_filename = env.back_inp_TS_filename
back_out_TS_filename = env.back_out_TS_filename
# defining paths for saving the trained NNs
checkpoint_path_TS_forw = env.checkpoint_path_TS_forw
checkpoint_path_TS_back = env.checkpoint_path_TS_back
# defining filenames for saving the mismatch edges per state-input pair
mismatch_list_forw_filename = env.mismatch_list_forw_filename
mismatch_list_back_filename = env.mismatch_list_back_filename
# defining filenames for saving the controller and winning domain
controller_filename = env.controller_filename
winning_domain_filename = env.winning_domain_filename



# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen()  # vector containing number of discrete points along each dimension in the
state_space_size = discrete_sys_size[0:dim_x]
input_space_size = discrete_sys_size[dim_x:dim_x+dim_u]

# state and input spaces
num_dis_states = np.prod(discrete_sys_size[0:dim_x]).astype(int)  # size of the state-space
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
num_state_inp_pairs = np.prod(discrete_sys_size).astype(int)  # number of state-input pairs
nn = 2 * (np.sum(discrete_sys_size[0:dim_x])+3*shift_no*dim_x)  # dimension of the vector at the output of the trained NN


# Run the function
make_keras_picklable()

# Load the input-output data
x_FW = np.memmap(forw_inp_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, dim_x+dim_u), offset=0)
y_FW = np.memmap(forw_out_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, nn), offset=0)
x_BW = np.memmap(back_inp_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, dim_x+dim_u), offset=0)
y_BW = np.memmap(back_out_TS_filename, dtype='float32', mode='r', shape=(num_state_inp_pairs, nn), offset=0)

# Load the model
NN_TS_forw = NN_structure_TS()
NN_TS_back = NN_structure_TS()
# Loads the weights
NN_TS_forw.load_weights(checkpoint_path_TS_forw)
NN_TS_back.load_weights(checkpoint_path_TS_back)
# Loading the computed list of misclassified state input pairs
#pairs_to_be_stored_list_forw_file = np.memmap(mismatch_list_forw_filename, dtype='float32', mode='r', shape=(35, dim_x + dim_u), offset=0)
pairs_to_be_stored_list_forw_file = np.memmap(mismatch_list_forw_filename, dtype='float32', mode='r', shape=(3, dim_x + dim_u), offset=0)

#pairs_to_be_stored_list_back_file = np.memmap(mismatch_list_back_filename, dtype='float32', mode='r', shape=(3, dim_x + dim_u), offset=0)
pairs_to_be_stored_list_back_file = np.memmap(mismatch_list_back_filename, dtype='float32', mode='r', shape=(0, dim_x + dim_u), offset=0)

"""pairs_to_be_stored_list_forw_file = np.load(mismatch_list_forw_filename, allow_pickle=True)
pairs_to_be_stored_list_back_file = np.load(mismatch_list_back_filename, allow_pickle=True)"""

Ctrl = np.memmap(controller_filename, dtype='float32', mode='w+', shape=(tuple(state_space_size)), offset=0)
# Compute the reachable set for the given goal_range
NN_FW_inp = np.zeros(dim_x+dim_u)
NN_BW_inp = np.zeros(dim_x+dim_u)
disc_FW = np.zeros((dim_x, 2))
disc_BW = np.zeros((dim_x, 2))
reach_set_ind = c2d_uapprox(Goal_range)
iter_reached = reach_set_ind.copy()
last_reach_new_ind = reach_set_ind.copy()
last_reach_old_ind = last_reach_new_ind.copy()
iter_no = 0
global_time = time.time()
while len(last_reach_new_ind) != 0:
    print(iter_no)
    print(len(last_reach_new_ind))
    start = time.time()
    ray.init(_plasma_directory="/tmp", log_to_driver=False)
    """y_FW_id = ray.put(y_FW[vec_ind_to_ind(state, state_space_size, dim_x)*num_dis_inputs, (1+vec_ind_to_ind(state, state_space_size, dim_x))*num_dis_inputs])
    y_BW_id = ray.put(y_BW[vec_ind_to_ind(state, state_space_size, dim_x)*num_dis_inputs, (1+vec_ind_to_ind(state, state_space_size, dim_x))*num_dis_inputs])"""
    Ctrl_id = ray.put(Ctrl)
    iter_reached_coded = [single_state_computation.remote(state, ray.put(y_FW[vec_ind_to_ind(state, state_space_size, dim_x)*num_dis_inputs: (1+vec_ind_to_ind(state, state_space_size, dim_x))*num_dis_inputs, :]),
                          ray.put(y_BW[vec_ind_to_ind(state, state_space_size, dim_x)*num_dis_inputs: (1+vec_ind_to_ind(state, state_space_size, dim_x))*num_dis_inputs, :]),
                          Ctrl_id) for state in last_reach_new_ind]
    iter_reached_decoded = ray.get(iter_reached_coded)
    del iter_reached_coded
    last_reach_new_ind = []
    print("length is", len(iter_reached_decoded))
    for i in range(len(iter_reached_decoded)):
        reachable_set_tuple = [tuple(item) for item in reach_set_ind]
        iter_reached_decoded_i = [tuple(item) for item in iter_reached_decoded[i]]
        iter_reached_decoded_squeezed_list = list(set(iter_reached_decoded_i).difference(set(reachable_set_tuple)))
        reach_set_ind.extend(iter_reached_decoded_squeezed_list)
        last_reach_new_ind.extend(iter_reached_decoded_squeezed_list)
    print("haaaa", time.time()-start, len(reach_set_ind))
    ray.shutdown()
    del iter_reached_decoded
    iter_no += 1

#  important ones: last_reach_new_ind, reach_set_ind, last_reach_old_ind
Ctrl.flush()  # save the controller

open_file = open(winning_domain_filename, "wb")
pickle.dump(reach_set_ind, open_file)
open_file.close()
print("The size of winning domain is", len(reach_set_ind))
print("The total synthesis time is", time.time()-global_time)




















