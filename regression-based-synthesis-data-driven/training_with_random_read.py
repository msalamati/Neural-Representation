"""By running this code, four Neural networks are trained and saved: two for estimating forward and backward end-points
 for each discrete state-input pairs; and two for estimating forward and backward growth bounds with respect
to discrete inputs."""
import numpy as np
import math
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import env_pool


def discrete_sys_size_gen(x_range):
    """ This function computes a vector that contains number of
     discrete states for every dimension of state and input spaces."""
    discrete_sys_size = np.zeros(dim_x + dim_u)
    for ii in range(0, dim_x):
        discrete_sys_size[ii] = math.floor((x_range[ii, 1] - x_range[ii, 0] - eta_x[ii]) / eta_x[ii] + 1)
    for ii in range(dim_x, dim_x + dim_u):
        discrete_sys_size[ii] = math.floor(
            (U_range[ii - dim_x, 1] - U_range[ii - dim_x, 0] - eta_u[ii - dim_x]) / eta_u[ii - dim_x] + 1)
    return discrete_sys_size.astype(int)


def NN_structure_TS():
    """Create the model for estimating the transition system."""
    model = Sequential()
    model.add(Dense(40, input_dim=dim_x+dim_u, activation=tf.nn.tanh))
    model.add(Dense(80, activation=tf.nn.tanh))
    model.add(Dense(35, activation=tf.nn.tanh))
    model.add(Dense(60, activation=tf.nn.tanh))
    model.add(Dense(dim_x, activation='linear'))
    return model

"""def transform_ind_BW_to_FW(ind_BW):
    # First compute the cell value corresponding to ind_BW w.r.t X_range
    pre_j = ind_BW
    sample_input = np.zeros(dim_x + dim_u)
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
    #  Now, compute the vectorized ind w.r.t the X_range_FW
    ind_vec = np.zeros(dim_x+dim_u)
    ind_vec[dim_x: dim_x+dim_u] = np.maximum(np.zeros(dim_u), np.minimum((sample_input[dim_x: dim_x+dim_u] - U_range[:, 0])//np.squeeze(eta_u), discrete_inp_size))
    ind_vec[0: dim_x] = np.maximum(np.zeros(dim_x), np.minimum((sample_input[0: dim_x] - X_range[:, 0]) // np.squeeze(eta_x), state_space_size_FW))
    # Finally, compute the global index in FW transition system
    ind_FW = 0
    for j in range(dim_x+dim_u):
        ind_FW += ind_vec[j] * np.prod(discrete_sys_size_FW[0:j])
    return int(ind_FW)"""

def My_Custom_Generator_TS(inp_filename, out_filename, num_samples, inp_dim, out_dim, batch_size):
    """On-the-fly data Generator for the training process (random-read)."""
    inputs = []
    targets = []
    batchcount = 0
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x = np.memmap(inp_filename, dtype='float32', mode='r+', shape=(num_samples, inp_dim), offset=0)
    y = np.memmap(out_filename, dtype='float32', mode='r+', shape=(num_samples, out_dim), offset=0)
    while True:
        for line in indices:
            linee = line
            inputs.append([x[linee, :]])
            targets.append([y[linee, :]])
            batchcount += 1
            if batchcount > batch_size:
                batch_x = np.array(inputs, dtype='float32')
                batch_y = np.array(targets, dtype='float32')
                yield (batch_x, batch_y)
                inputs = []
                targets = []
                batchcount = 0


def My_Custom_Generator_seq_read(inp_filename, out_filename):
    """On-the-fly data Generator for the training process (sequential-read)."""
    inputs = []
    targets = []
    batchcount = 0
    num_loads = num_state_inp_pairs//chunck_size+1
    indices = np.arange(chunck_size)
    np.random.shuffle(indices)
    while True:
        for i in range(num_loads):
            num_samples_to_load = min(chunck_size, num_state_inp_pairs-i*chunck_size)
            if num_samples_to_load != chunck_size:
                indices = np.arange(num_samples_to_load)
                np.random.shuffle(indices)
            x_buffer = np.memmap(inp_filename, dtype='float32', mode='r+', shape=(num_samples_to_load, dim_x + dim_u), offset=int(i*chunck_size * (dim_x + dim_u) * 32 / 8))
            y_buffer = np.memmap(out_filename, dtype='float32', mode='r+', shape=(num_samples_to_load, dim_x), offset=int(i*chunck_size * (dim_x) * 32 / 8))
            ind = 0
            for line in indices:
                # for line in range(num_batches):
                x = x_buffer[line, 0:dim_x + dim_u]
                y = y_buffer[line, 0:dim_x]
                inputs.append(x)
                targets.append(y)
                batchcount += 1
                ind += 1
                if batchcount > batch_size or ind == chunck_size:
                    batch_x = np.array(inputs, dtype='float32')
                    batch_y = np.array(targets, dtype='float32')
                    yield (batch_x, batch_y)
                    inputs = []
                    targets = []
                    batchcount = 0

# Setting the parameters
env = env_pool.inv_pend()

X_range = env.X_range  # state-space
# X_range_FW = env.X_range_FW  # state-space for the FW computations
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
# learning related settings
epochs_TS = env.epochs_TS
epochs_GB = env.epochs_GB
batch_size_TS = env.batch_size_TS
batch_size_GB = env.batch_size_GB
learning_rate = env.learning_rate
# defining filenames for saving the transition system
forw_sub_inp_TS_filename = env.forw_sub_inp_TS_filename
forw_sub_out_TS_c_filename = env.forw_sub_out_TS_c_filename
forw_sub_out_TS_r_filename = env.forw_sub_out_TS_r_filename
"""forw_sub_inp_TS_filename = env.forw_inp_TS_filename
forw_sub_out_TS_c_filename = env.forw_out_TS_c_filename
forw_sub_out_TS_r_filename = env.forw_out_TS_r_filename"""
back_inp_TS_filename = env.back_inp_TS_filename
back_out_TS_c_filename = env.back_out_TS_c_filename
back_out_TS_r_filename = env.back_out_TS_r_filename


# defining paths for saving the trained NNs
checkpoint_path_TS_forw = env.checkpoint_path_TS_forw
checkpoint_path_TS_back = env.checkpoint_path_TS_back
checkpoint_path_GB_forw = env.checkpoint_path_GB_forw
checkpoint_path_GB_back = env.checkpoint_path_GB_back






# Extract descriptive parameters of the system
rr_x = eta_x / 2  # radius of the partitions in the state-space
rr_u = eta_u / 2  # radius of the partitions in the input-space
dim_x = np.shape(X_range)[0]  # dimension of the state-space
dim_u = np.shape(U_range)[0]  # dimension of the input-space
discrete_sys_size = discrete_sys_size_gen(X_range)  # vector containing number of discrete pointsalong each dimension in the
discrete_inp_size = discrete_sys_size[dim_x:dim_x+dim_u]
state_space_size = discrete_sys_size[0:dim_x]
discrete_sys_size_FW = discrete_sys_size_gen(X_range)
state_space_size_FW = discrete_sys_size_FW[0:dim_x]

# state and input spaces
num_dis_states = np.prod(discrete_sys_size[0:dim_x]).astype(int)  # size of the state-space
num_dis_inputs = np.prod(discrete_sys_size[dim_x:dim_x + dim_u]).astype(int)  # size of the input-space
num_state_inp_pairs = np.prod(discrete_sys_size).astype(int)  # number of state-input pairs
num_state_inp_pairs_FW = np.prod(discrete_sys_size_FW).astype(int)  # number of state-input pairs

# Training using TF 2.x
num_batches_TS = int(np.ceil(np.prod(discrete_sys_size).astype(int) / batch_size_TS))
num_batches_GB = int(np.ceil(np.prod(discrete_sys_size[dim_x:dim_x+dim_u]).astype(int) / batch_size_GB))
# chunck_size = batch_size_TS * 100

# Compile the model
NN_TS_forw = NN_structure_TS()
NN_TS_back = NN_structure_TS()
NN_GB_forw = NN_structure_TS()
NN_GB_back = NN_structure_TS()

# Training using TF 2.x
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=math.ceil(num_state_inp_pairs/num_batches_TS),
    decay_rate=.999,
    staircase=True)
NN_TS_forw.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.999, beta_2=.99999, name='Adam'),
               metrics=['mean_squared_error'])
NN_TS_back.compile(loss='mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.999, beta_2=.99999, name='Adam'),
               metrics=['mean_squared_error'])
NN_GB_forw.compile(loss='mean_squared_error',
           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.999, beta_2=.99999, name='Adam'),
           metrics=['mean_squared_error'])
NN_GB_back.compile(loss='mean_squared_error',
           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.999, beta_2=.99999, name='Adam'),
           metrics=['mean_squared_error'])
# Create a callback that saves the model's weights
cp_callback_TS_forw = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_TS_forw,
                                                 save_weights_only=True,
                                                 verbose=1)
cp_callback_TS_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_TS_back,
                                                 save_weights_only=True,
                                                 verbose=1)

cp_callback_GB_forw = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_GB_forw,
                                             save_weights_only=True,
                                             verbose=1)
cp_callback_GB_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_GB_back,
                                             save_weights_only=True,
                                             verbose=1)
# Fit data to models using data written on disk+save the trained models
start = time.time()
history1 = NN_TS_forw.fit(My_Custom_Generator_TS(forw_sub_inp_TS_filename, forw_sub_out_TS_c_filename, num_state_inp_pairs, dim_x+dim_u, dim_x, batch_size_TS),
                     steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_TS_forw])
print("Execution time for FW training", time.time() - start)
history2 = NN_TS_back.fit(My_Custom_Generator_TS(back_inp_TS_filename, back_out_TS_c_filename, num_state_inp_pairs, dim_x+dim_u, dim_x, batch_size_TS),
                     steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_TS_back])
print("Execution time for FW+BW training", time.time() - start)

history3 = NN_GB_forw.fit(My_Custom_Generator_TS(forw_sub_inp_TS_filename, forw_sub_out_TS_r_filename, num_state_inp_pairs, dim_x+dim_u, dim_x, batch_size_TS),
                 steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_GB_forw])
history4 = NN_GB_back.fit(My_Custom_Generator_TS(back_inp_TS_filename, back_out_TS_r_filename, num_state_inp_pairs, dim_x+dim_u, dim_x, batch_size_TS),
                 steps_per_epoch=num_batches_TS, epochs=epochs_TS, verbose=1, validation_split=0, callbacks=[cp_callback_GB_back])
print("Execution time for full training", time.time() - start)