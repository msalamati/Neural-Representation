"""By running this code, the (size for) list of states for which the neural network
does not calculate a valid control input is computed."""
import os
import numpy as np
import math
import time
import scipy.io
import ray
import mat73
import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import env_pool

# Defining the functions


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



"""def NN_structure():
    model = Sequential()
    model.add(Dense(20, input_dim=dim_x, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.tanh))
    model.add(Dense(160, activation=tf.nn.tanh))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(250, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(num_dis_inputs, activation='linear'))
    return model"""

def NN_structure():
    """Create the model for guessing valid control inputs."""
    model = Sequential()
    model.add(Dense(20, input_dim=dim_x, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(80, activation=tf.nn.relu))
    model.add(Dense(160, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))

    """model.add(Dense(250, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))
    model.add(Dense(200, activation=tf.nn.relu))"""
    model.add(Dense(num_dis_inputs, activation='linear'))
    return model

def approximation_error():
    counter = 0
    for step in range(win_domain_size_not_scaled // num_tasks_per_step+1):
        offset = step*num_tasks_per_step
        num_tasks_in_this_step = min(num_tasks_per_step, win_domain_size_not_scaled - step * num_tasks_per_step)
        #start = time.time()
        e_coded = [single_point_error_TS.remote(step * num_tasks_per_step // length + i, num_tasks_in_this_step-i*length, win_domain_coded, controller_coded, NN_controller_coded, num_valid_cont_list_coded) for i in range(0, num_tasks_in_this_step // length+1)]
        e_decoded = ray.get(e_coded)
        #  print('here is the time', time.time()-start)
        for i in range(num_tasks_in_this_step // length+1):
            counter += len(e_decoded[i])
        #print("time is", time.time()-start)
    return counter

@ray.remote#(num_returns=2)
def single_point_error_TS(ii, num_tasks, win_domain_codedd, controller_codedd, NN_controller_codedd, num_valid_cont_list_codedd):
    length_for_this_step = min(length, num_tasks)
    to_be_saved_list = []
    bin_len = dim_x
    for k in range(length_for_this_step):
        state_main = win_domain_codedd[ii * length + k, :]
        row_ind_set = np.where(np.all(controller_codedd[:, 0:dim_x] == win_domain_codedd[ii*length+k, :], axis=1))[0]
        count = len(row_ind_set)  # num_valid_cont_list_codedd[ii*length+k].astype(int)
        for sub_cell_idx in range(scale**dim_x):
            bin_sub_cell_idx = binary_converter(sub_cell_idx, bin_len)
            sub_cell_center = state_main - np.squeeze(eta_x) * scale/2 + np.squeeze(eta_x)/2+ bin_sub_cell_idx*np.squeeze(eta_x)
            NN_inp = sub_cell_center
            NN_out = NN_controller_codedd(np.array([NN_inp]))
            NN_argmax_ind = np.argmax(NN_out)
            for j in range(count):
                inp_exact_ind = cell_to_ind(controller_codedd[row_ind_set[j], dim_x:dim_x + dim_u]).astype(int)
                valid = 0
                if NN_argmax_ind == inp_exact_ind:
                    valid = 1
                    break
            if valid == 0:
                #  print("We need to save this one!")
                to_be_saved_list.append(copy.copy(sub_cell_center))
    return to_be_saved_list


def cell_to_ind(vec):
    vectorized_ind = np.maximum(np.zeros(dim_u), np.minimum((vec - U_range[:, 0])//np.squeeze(eta_u), input_space_size))
    ind = vectorized_ind[0]
    for ii in range(1, dim_u):
        ind += vectorized_ind[ii]*np.prod(input_space_size[0:ii-1])
    return ind


def binary_converter(x, bits):
    """function to convert a number x into a vector of bits"""
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])


"""def ind_to_cell(ind, full_range, space_size, eta, dim):
    cell_center = np.zeros(dim)
    rr = eta / 2
    pre_j = ind
    for j in (range(dim)):
        if j == dim:
            ind_j = pre_j
        else:
            ind_j = pre_j // np.prod(space_size[0:j])
            pre_j -= ind_j * np.prod(space_size[0:j])
        cell_center[j] = full_range[j, 0] + rr[j] + eta[j] * ind_j

    return cell_center"""

# Setting the parameters
env = env_pool.tora()

X_range = env.X_range  # state-space
U_range = env.U_range  # input space
sample_time = env.sample_time  # sampling time in seconds
scale = env.scale
eta_x = env.eta_x/scale  # state-space discretization size
eta_u = env.eta_u  # input-space discretization size
# parallelization parameters
length = env.length
num_tasks_per_step = env.num_tasks_per_step
# learning related settings
epochs = env.epochs
batch_size = env.batch_size
learning_rate = env.learning_rate
# defining filenames for loading the controller
win_domain_filename_not_scaled = env.win_domain_filename_not_scaled
win_domain_filename_scaled = env.win_domain_filename_scaled
controller_filename_not_scaled = env.controller_filename_not_scaled
# defining filenames for the processed output data
onehot_filename = env.onehot_filename
num_valid_cont_list_filename = env.num_valid_cont_list_filename
# defining paths for saving the trained NNs
checkpoint_path_NN_controller = env.checkpoint_path_NN_controller


# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen()  # vector containing number of discrete pointsalong each dimension in the
state_space_size = discrete_sys_size[0:dim_x]
input_space_size = discrete_sys_size[dim_x:dim_x+dim_u]
# state and input spaces
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
# winning domain size
win_domain_not_scaled = scipy.io.loadmat(win_domain_filename_not_scaled)["win_domain"]
#win_domain_not_scaled = mat73.loadmat(win_domain_filename_not_scaled)['win_domain']
win_domain_size_not_scaled = win_domain_not_scaled.shape[0]
win_domain_size_scaled = win_domain_size_not_scaled * (scale**dim_x)
win_domain_scaled = np.memmap(win_domain_filename_scaled, dtype='float32', mode='r', shape=(win_domain_size_scaled, dim_x))

controller_not_scaled = scipy.io.loadmat(controller_filename_not_scaled)["controller"]
#controller_not_scaled = mat73.loadmat(controller_filename_not_scaled)['controller']
onehot = np.memmap(onehot_filename, dtype='float32', mode='r', shape=(win_domain_size_scaled, num_dis_inputs))  #  np.zeros((win_domain_size, num_dis_inputs))
num_valid_cont_list = np.memmap(num_valid_cont_list_filename, dtype='float32', mode='r', shape=(win_domain_size_scaled))


# Run the function
make_keras_picklable()
# Compile the model
NN_controller = NN_structure()
# Loads the weights
NN_controller.load_weights(checkpoint_path_NN_controller)



ray.init(_plasma_directory="/tmp", log_to_driver=False)
NN_controller_coded = ray.put(NN_controller)
controller_coded = ray.put(controller_not_scaled)
num_valid_cont_list_coded = ray.put(num_valid_cont_list)
win_domain_coded = ray.put(win_domain_not_scaled)
start = time.time()
count = approximation_error()
print("the portion of misclassified states is", count/win_domain_size_scaled)
print("The execution time is:", time.time()-start)

